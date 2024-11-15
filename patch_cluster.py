import math
import pickle as pk
import pathlib

import torch
import numpy as np
import scipy.sparse.csgraph as scs
import scipy.sparse as scpr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors
import kmedoids
import seaborn as sns
import matplotlib.pyplot as plt

from data import MET_Data
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

merge_map = utils.get_tree_merge_map("data/meta/tree_Mouse_ALM-VISp_2018.csv", "n3")

def get_merge_func(index, merge_map = merge_map):
    merge_func = np.vectorize(lambda elem: merge_map[elem][index]) 
    return merge_func

def get_raw_distance(raw_data, source_i, target_i, **kwargs):
    return get_euclidean_distance(raw_data, source_i, target_i)

def get_euclidean_distance(z_means, source_i, target_i, **kwargs):
    diff_vectors = z_means[source_i, None] - z_means[target_i]
    dist = torch.linalg.norm(diff_vectors, dim = -1)
    return dist

def get_sphered_distance(z_means, z_transf, source_i, target_i, **kwargs):
    diff_vectors = z_means[source_i, None] - z_means[target_i]
    cov = torch.einsum("nij,nkj->nik", z_transf, z_transf)
    scales = torch.linalg.eigh(cov)[0][:, 0]
    dist = torch.sqrt(torch.square(diff_vectors).sum(-1) / scales)
    return dist

def get_cov_distance(z_means, z_transf, num_steps, source_i, target_i, cov_reg = 1e-10, **kwargs):
    (source_means, source_transf_1, source_transf_2) = (z_means[source_i], z_transf[0][source_i], z_transf[1][source_i])
    (nn_means, nn_transf_1, nn_transf_2) = (z_means[target_i], z_transf[0][target_i], z_transf[1][target_i])
    cov_reg_tensor = cov_reg*torch.diag_embed(torch.ones_like(source_means))
    diff_vectors = nn_means - source_means[:, None]

    steps = torch.linspace(0, 1, num_steps + 1)[1:].to(device)
    source_inv = torch.linalg.inv(torch.einsum("nij,nkj->nik", source_transf_1, source_transf_1) + torch.einsum("nij,nkj->nik", source_transf_2, source_transf_2) + cov_reg_tensor)
    target_inv = torch.linalg.inv(torch.einsum("nlij,nlkj->nlik", nn_transf_1, nn_transf_1) + torch.einsum("nlij,nlkj->nlik", nn_transf_2, nn_transf_2) + cov_reg_tensor[:, None])
    interp_invs = torch.einsum("s,nlij->snlij", 1 - steps, source_inv[:, None]) + torch.einsum("s,nlij->snlij", steps, target_inv)
    dist = torch.sqrt(torch.einsum("nli,snlij,nlj->snl", diff_vectors, interp_invs, diff_vectors)).mean(0)
    return dist

def get_riemann_distance(z_means, metric, num_steps, source_i, target_i, **kwargs):
    steps = torch.linspace(0, 1, num_steps + 1).to(device)
    diff_vectors = z_means[target_i] - z_means[source_i, None]
    step_vectors = z_means[None, source_i, None] + torch.einsum("s,nli->snli", steps, diff_vectors)
    with torch.no_grad():
        step_transf = metric(step_vectors.flatten(0, 2).float()).unflatten(0, step_vectors.shape[:3])
    step_inv = torch.linalg.inv(torch.einsum("snlij,snlkj->snlik", step_transf, step_transf))
    mean_inv = step_inv[1:] #(step_inv[1:] + step_inv[:-1]) / 2
    dist = torch.sqrt(torch.einsum("nli,snlij,nlj->snl", diff_vectors, mean_inv, diff_vectors)).mean(0)
    return dist

def get_decoded_distance(z_means, decoder, num_steps, source_i, target_i, dec_batch_size = 1024, **kwargs):
    steps = torch.linspace(0, 1, num_steps + 1).to(device)
    diff_vectors = z_means[target_i] - z_means[source_i, None]
    step_vectors = z_means[None, source_i, None] + torch.einsum("s,nli->snli", steps, diff_vectors)
    flat_norms = []
    flat_vectors = step_vectors.flatten(1, 2).float()
    num_batches = math.ceil(flat_vectors.shape[1] / dec_batch_size)
    for batch in (flat_vectors[:, i*dec_batch_size:(i+1)*dec_batch_size] for i in range(num_batches)):
        with torch.no_grad():
            step_recon = next(iter(decoder(batch.flatten(0, 1)).values())).unflatten(0, batch.shape[:2]).flatten(2)
            flat_norms.append(torch.linalg.norm(step_recon[1:] - step_recon[:-1], dim = -1).sum(0))
    dist = torch.cat(flat_norms, 0).unflatten(0, step_vectors.shape[1:3])
    return dist

def get_hop_distance(z_means, decoder, source_i, target_i, **kwargs):
    return get_decoded_distance(z_means, decoder, 1, source_i, target_i)

def get_distances(metric_func, z_means, batch_size = 128, distance = True, **kwargs):
    num_samples = z_means.shape[0]
    batches = []
    for i in range(math.ceil(num_samples / batch_size)):
        source_i = torch.arange(i*batch_size, min((i+1)*batch_size, num_samples))
        dist = metric_func(z_means = z_means, source_i = source_i, target_i = torch.arange(num_samples)[None], **kwargs)
        if not distance:
            dist = 1 / dist
        batches.append(dist)
    dist_matrix = torch.cat(batches)
    distances = (dist_matrix.T + dist_matrix) / 2
    return distances

def get_tree_labels(tree_children, true_labels, merge_list):
    num_samples = len(true_labels)
    parent_map = [{child_1: i, child_2: i} for (i, (child_1, child_2)) in enumerate(tree_children, num_samples)]
    cluster_indices = [np.arange(num_samples)]
    for merge_dict in parent_map:
        cluster_indices.append(np.asarray([merge_dict.get(i, i) for i in cluster_indices[-1]]))
    pred_clusters = []
    for label_merge in merge_list:
        target_labels = get_merge_func(label_merge)(true_labels)
        label_map = {label:i for (i, label) in enumerate(np.unique(target_labels))}
        pred_clusters.append(cluster_indices[-len(label_map)])
    return pred_clusters

def get_kmedoid_labels(results, modals, fold, true_labels, distances, merge_list):
    for merges in merge_list:
        print(f"{modals[0]} -> {modals[1]} ({fold}) -- Merge {merges}                                                      ", end = "\r")
        target_labels = get_merge_func(merges)(true_labels)
        label_map = {label:i for (i, label) in enumerate(np.unique(target_labels))}
        num_clusters = len(label_map)
        pred_labels = kmedoids.KMedoids(num_clusters, random_state = 42).fit(distances).labels_
        results.setdefault(modals, {}).setdefault(merges, {})[fold] = pred_labels
    return results

def get_agglomerative_labels(results, modals, fold, true_labels, distances, merge_list):
    model = AgglomerativeClustering(metric = "precomputed", linkage = "average", compute_full_tree = True).fit(distances)
    pred_labels = get_tree_labels(model.children_, true_labels, merge_list)
    for (merges, merge_labels) in zip(merge_list, pred_labels):
        results.setdefault(modals, {}).setdefault(merges, {})[fold] = merge_labels
    return results

def get_raw_labels(results, modals, fold, cluster_labels, raw_data, merge_list):
    model = AgglomerativeClustering(linkage = "complete", compute_full_tree = True).fit(raw_data.reshape(raw_data.shape[0], -1))
    pred_labels = get_tree_labels(model.children_, cluster_labels, merge_list)
    for (merges, merge_labels) in zip(merge_list, pred_labels):
        results.setdefault(modals, {}).setdefault(merges, {})[fold] = merge_labels
    return results

def get_shortest_distances(distance_matrix, num_neighbors):
    neighborhood = NearestNeighbors(n_neighbors = num_neighbors, metric = "precomputed").fit(distance_matrix.numpy(force = True))
    sparse_distances = neighborhood.kneighbors_graph(mode = "distance")
    sym_distances = (sparse_distances + sparse_distances.T) / 2
    distances = np.nan_to_num(scs.shortest_path(sym_distances))
    return torch.from_numpy(distances).to(distance_matrix.device)

def get_nearest_neighbor_distances(distance_matrix, means, num_neighbors):
    neighborhood = NearestNeighbors(n_neighbors = num_neighbors).fit(means.numpy(force = True))
    connectivity = neighborhood.kneighbors_graph(mode = "connectivity").toarray()
    sparse_distances = scpr.coo_array(distance_matrix.numpy(force = True) * connectivity)
    sparse_distances.eliminate_zeros()
    sym_distances = (sparse_distances + sparse_distances.T) / 2
    distances = np.nan_to_num(scs.shortest_path(sym_distances))
    return torch.from_numpy(distances).to(distance_matrix.device)

def get_model(model_path):
    experiment = utils.load_jit_folds(model_path, get_checkpoints = False)
    encoders = {fold: {modal: arms["enc"].to(device) for (modal, arms) in fold_dict["best"].model_dict.items()}
                for (fold, fold_dict) in experiment["folds"].items()}
    decoders = {fold: {modal: arms["dec"].to(device) for (modal, arms) in fold_dict["best"].model_dict.items()}
                for (fold, fold_dict) in experiment["folds"].items()}
    mappers = {fold: {tuple(modal_str.split("-")): mapper.to(device) for (modal_str, mapper) in fold_dict["best"].mappers.items()}
                     if fold_dict["best"].mappers else {}
               for (fold, fold_dict) in experiment["folds"].items()}
    return (encoders, decoders, mappers)

def get_data():
    data_keys = {"logcpm": "logcpm", "pca-ipfx": "pca-ipfx", "arbors": "arbors"}
    met_data = MET_Data("data/neurons.hdf5", **data_keys)
    for form in ["logcpm", "pca-ipfx", "arbors"]:
        met_data.cache_data(form)
    raw_data = met_data.query(formats = [("logcpm", "pca-ipfx", "arbors")], outputs = ["logcpm", "pca-ipfx", "arbors", "cluster_label"])
    cluster_labels = np.char.strip(raw_data["cluster_label"])
    data = {form: torch.from_numpy(raw_data[form]).float().to(device) for form in ["logcpm", "pca-ipfx", "arbors"]}
    return (data, cluster_labels)

def get_model_outputs(data, encoders, decoders, mappers, forms):
    with torch.no_grad():
        modal_latent = {fold: {modal: encoder({forms[modal]:data[forms[modal]]}) for (modal, encoder) in fold_dict.items()} 
                        for (fold, fold_dict) in encoders.items()}
        mapped_recons = {fold: {(in_m, out_m): decoders[fold][out_m](mapper(modal_latent[fold][in_m][0])[0])[forms[out_m]] 
                                for ((in_m, out_m), mapper) in fold_dict.items()}
                        for (fold, fold_dict) in mappers.items()} 
        direct_recons = {fold: {(in_m, in_m): decoder(modal_latent[fold][in_m][0])[forms[in_m]] 
                                for (in_m, decoder) in fold_dict.items()}
                        for (fold, fold_dict) in decoders.items()}
        recons = {fold: {**direct_recons[fold], **mapped_recons[fold]} for fold in direct_recons}

    (means, transfs, eig_vals, eig_vecs, rotations) = ({}, {}, {}, {}, {})
    with torch.no_grad():
        for (fold, fold_dict) in recons.items():
            for (in_modal, out_modal) in fold_dict:
                (mean, transf) = modal_latent[fold][in_modal]
                modals = (in_modal, out_modal)
                if in_modal != out_modal:
                    (mean, transf) = mappers[fold][modals](mean)
                (means.setdefault(fold, {})[modals], transfs.setdefault(fold, {})[modals]) = (mean, transf)
                covs = torch.einsum("nij,nkj->nik", transf, transf)
                (eig_vals.setdefault(fold, {})[modals], eig_vecs.setdefault(fold, {})[modals]) = torch.linalg.eigh(covs)
                rotations.setdefault(fold, {})[modals] = 180*torch.atan2(eig_vecs[fold][modals][:, 1, 1], eig_vecs[fold][modals][:, 1, 0])/np.pi
    return (means, transfs, recons)

def run(file_path, model_path, method_funcs, method_params, merge_list, dijkstra = True, **kwargs):
    import warnings
    warnings.filterwarnings('ignore')

    forms = {"T": "logcpm", "E": "pca-ipfx", "M": "arbors"}
    clustering_funcs = {"medoid": get_kmedoid_labels, "tree": get_agglomerative_labels}
    (encoders, decoders, mappers) = get_model(model_path)
    (data, cluster_labels) = get_data()
    (means, transfs, recons) = get_model_outputs(data, encoders, decoders, mappers, forms)
    
    results = {method: {} for method in method_funcs}
    for (fold, recon_dict) in recons.items():
        for modals in recon_dict:
            for (method, func) in method_funcs.items():
                print(f"{modals[0]} -> {modals[1]} ({fold}/{len(recons)}) -- Computing {method} distances                       ", end = "\r")
                if method == "raw":
                    raw_data = data[forms[modals[1]]].numpy(force = True)
                    results[method] = get_raw_labels(results[method], modals, fold, cluster_labels, raw_data, merge_list)
                else:
                    clustering_func = clustering_funcs[method_params[method]["cluster_alg"]]
                    (modal_mean, modal_transf) = (means[fold][modals], transfs[fold][modals])
                    modal_transf = (modal_transf, transfs[fold][(modals[0], modals[0])])
                    distances = get_distances(func, modal_mean, z_transf = modal_transf, decoder = decoders[fold][modals[1]], **method_params[method])
                    if dijkstra:
                        distances = get_nearest_neighbor_distances(distances, modal_mean, 10)
                    results[method] = clustering_func(results[method], modals, fold, cluster_labels, distances.numpy(force = True), merge_list)
        with open(file_path, "wb") as target:
            pk.dump({"pred_labels": results, "true_labels": cluster_labels}, target)

def plot(file_path, score_func, graph_text, score_label):
    with open(file_path, "rb") as target:
        results = pk.load(target)
    (pred_labels, true_labels) = (results["pred_labels"], results["true_labels"])
    plot_dict = {"Method": [], "Input Modality": [], "Output Modality": [], "Merge": [], "Fold": [], "Rand Score": []}
    for (method, method_dict) in pred_labels.items():
        if method == "hop": continue
        for ((in_modal, out_modal), modal_dict) in method_dict.items():
            for (merge, merge_dict) in modal_dict.items():
                print(f"Generating {method} scores: {in_modal} -> {out_modal}                     ", end = "\r")
                true_merged = get_merge_func(merge)(true_labels)
                label_map = {label:i for (i, label) in enumerate(np.unique(true_merged))}
                true_indices = np.asarray([label_map[label] for label in true_merged])
                for (fold, pred_labels) in merge_dict.items():
                    plot_dict["Method"].append(method)
                    plot_dict["Input Modality"].append(in_modal)
                    plot_dict["Output Modality"].append(out_modal)
                    plot_dict["Merge"].append(merge)
                    plot_dict["Fold"].append(fold)
                    plot_dict[score_label].append(score_func(true_indices, pred_labels))
    print(f"Generating plot for {pathlib.Path(file_path).stem}                                  ")
    plot = sns.relplot(data = plot_dict, x = "Merge", y = score_label, hue = "Method", row = "Input Modality", col = "Output Modality", kind = "line")
    for ax in plot.figure.axes:
        ax.set_xlim((0, 103))
        ax.set_ylim((0.0, 1.1))
    plot.figure.axes[0].text(10, 0.9, graph_text, {"fontsize": 40})
    plt.savefig(f"{pathlib.Path(file_path).stem}.png", bbox_inches = "tight")
    plt.close()

args = [
    {"model_path": "results/baselines/met_10d_mse", 
     "file_path": "data/clustering/cluster_t_summed.pk",
     "title": "Summed",
     "method_funcs": {
        "cov": get_cov_distance, 
        "jac": get_decoded_distance, 
        "hop": get_hop_distance, 
        "eucl": get_euclidean_distance, 
        "raw": None},
     "method_params": {
         "cov": {"num_steps": 10, "cluster_alg": "tree", "distance": True, "batch_size": 128},
         "jac": {"num_steps": 10, "cluster_alg": "tree", "distance": True, "batch_size": 128},
         "hop": {"cluster_alg": "tree", "distance": True},
         "eucl": {"cluster_alg": "tree", "distance": True},
        },
    "merge_list": list(range(len(next(iter(merge_map.values()))) - 1)),
    "dijkstra": True}
    ]

for kwargs in args:
    run(**kwargs)
    # plot(kwargs["file_path"], adjusted_rand_score, graph_text = kwargs["title"], score_label = "Rand Score")