import contextlib

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def align(array, specimen_ids, anno):
    aligned_array = np.stack([np.full_like(array[0], np.nan)] * len(anno))
    id_df = pd.DataFrame(data = {"specimen_id": specimen_ids, "alignment_index": range(len(specimen_ids))})
    id_df["specimen_id"] = id_df["specimen_id"].str.strip()
    aligned_indices = pd.merge(anno, id_df, on = "specimen_id", how = "inner")
    aligned_array[aligned_indices["row_index"].to_numpy()] =  array[aligned_indices["alignment_index"].to_numpy()]
    return aligned_array

def select(array, anno):
    selected = array[anno["row_index"]]
    return selected

def normalize_sample_count(exp1, exp2, anno):
    unnormalized = anno.query("platform == @exp1 or platform == @exp2")
    exp1_exc = unnormalized.query("platform == @exp1 and `class` == 'exc'")
    exp1_inh = unnormalized.query("platform == @exp1 and `class` == 'inh'")
    exp2_exc = unnormalized.query("platform == @exp2 and `class` == 'exc'")
    exp2_inh = unnormalized.query("platform == @exp2 and `class` == 'inh'")
    if len(exp1_exc) > len(exp2_exc):
       exp1_exc = exp1_exc.sample(len(exp2_exc), replace = False)
    else:
        exp2_exc = exp2_exc.sample(len(exp1_exc), replace = False)
    if len(exp1_inh) > len(exp2_inh):
       exp1_inh = exp1_inh.sample(len(exp2_inh), replace = False)
    else:
        exp2_inh = exp2_inh.sample(len(exp1_inh), replace = False)
    anno_samp = pd.concat([exp1_exc, exp1_inh, exp2_exc, exp2_inh])
    return (unnormalized, anno_samp)

def load_raw_data():
    anno_full = pd.read_csv("data/raw/exc_inh_ME_fMOST_EM_specimen_ids_shuffled_4Apr23.txt").rename(columns = {"Unnamed: 0": "row_index"})
    anno_full["specimen_id"] = anno_full["specimen_id"].str.strip()
    arbor_dict = sio.loadmat("data/raw/M_arbor_data_50k_4Apr23.mat")
    arbors = align(arbor_dict["hist_ax_de_api_bas"], arbor_dict["specimen_id"], anno_full)
    anno = anno_full.query("M_cell")
    (exp1, exp2) = ("EM", "patchseq")
    (anno_unnorm, anno_samp) = normalize_sample_count(exp1, exp2, anno)
    anno_exc = anno_unnorm.query("`class` == 'exc'")
    em_arbors = select(arbors, anno_exc.query("platform == 'EM'"))[..., 2:].reshape((-1, 960))
    patch_arbors = select(arbors, anno_exc.query("platform == 'patchseq'"))[..., 2:].reshape((-1, 960))
    return (em_arbors, patch_arbors)

def ridge_regression(X_em, X_patch, alpha, fitting_gradient = False):
    fitting_context = torch.no_grad() if not fitting_gradient else contextlib.nullcontext()
    with fitting_context:
        y = torch.cat([-torch.ones(X_em.shape[0]), torch.ones(X_patch.shape[0])]).to(X_em)
        X = torch.cat([X_em, X_patch])
        X_offset = torch.mean(X, dim = 0, keepdim = True)
        (U, s, Vt) = torch.linalg.svd(X - X_offset, full_matrices = False)
        s_reg = s / (s**2 + alpha)
        coeff = torch.einsum("ij,jk,k->i", Vt.transpose(0, 1)*s_reg, U.transpose(0, 1), y)
        intercept = torch.mean(y)
    return (X_offset, intercept, coeff)

class Aligner(torch.nn.Module):
    def __init__(self, hidden_dims, activation):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.layer_names = []
        all_dims = [960] + list(hidden_dims) + [960]
        for (i, (in_dim, out_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:]), 1):
            setattr(self, f"l{i}", torch.nn.Linear(in_dim, out_dim))
            self.layer_names.append(f"l{i}")
        self.relu = torch.nn.ReLU()
        if activation == "linear":
            self.activation = torch.nn.Identity()
        elif activation == "relu":
            self.activation = torch.nn.functional.relu
        elif activation == "softplus":
            self.activation = torch.nn.functional.softplus
        else:
            raise ValueError(f'Activation "{activation}" not recognized.')

    def forward(self, X):
        for layer_name in self.layer_names[:-1]:
            layer = getattr(self, layer_name)
            X = self.relu(layer(X))
        final_layer = getattr(self, self.layer_names[-1])
        output = self.activation(final_layer(X))
        return output
    
class Classifier(torch.nn.Module):
    def __init__(self, hidden_dims, gauss_noise):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.gauss_noise = gauss_noise
        self.layer_names = []
        all_dims = [960] + list(hidden_dims) + [1]
        for (i, (in_dim, out_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:]), 1):
            setattr(self, f"l{i}", torch.nn.Linear(in_dim, out_dim))
            self.layer_names.append(f"l{i}")
        self.relu = torch.nn.ReLU()

    def get_loss(self, X):
        log_odds = self(X)[:, 0]
        labels = torch.zeros(X.shape[0]).to(X.device).long()
        loss = torch.nn.functional.cross_entropy(torch.stack([-log_odds, log_odds], 1), labels)
        return loss

    def forward(self, X):
        X = X + self.gauss_noise*torch.randn_like(X)
        for layer_name in self.layer_names[:-1]:
            layer = getattr(self, layer_name)
            X = self.relu(layer(X))
        final_layer = getattr(self, self.layer_names[-1])
        output = final_layer(X)
        return output
    
class ExcDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, device):
        super().__init__()
        self.X = torch.from_numpy(X).float().to(device)
        self.y = torch.from_numpy(y).to(device)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


class ArborDataset(torch.utils.data.IterableDataset):
    def __init__(self, em_arbors, patch_arbors, device):
        super().__init__()
        self.em_arbors = torch.from_numpy(em_arbors).float().to(device)
        self.patch_arbors = torch.from_numpy(patch_arbors).float().to(device)

    def __iter__(self):
        patch_indices = torch.randperm(len(self.patch_arbors))
        em_indices = torch.randperm(len(self.em_arbors))
        patch_step = 0
        for em_index in em_indices:
            if patch_step == len(self.patch_arbors):
                patch_indices = torch.randperm(len(self.patch_arbors))
                patch_step = 0
            patch_index = patch_indices[patch_step]
            arbors = torch.stack([self.em_arbors[em_index], self.patch_arbors[patch_index]], 0)
            yield arbors 
            patch_step +=1         

class EarlyStopping():
    def __init__(self, exp_dir, patience, min_improvement_fraction):
        self.exp_dir = exp_dir
        self.patience = patience
        self.frac = min_improvement_fraction
        self.counter = 0
        self.best_epoch = 0
        self.min_loss = np.inf

    def stop_check(self, loss, model, epoch):
        if loss < (1 - self.frac) * self.min_loss:
            self.counter = 0
            self.min_loss = loss
            torch.save(model.state_dict(), self.exp_dir / f"best_params.pt")
            self.best_epoch = epoch
        else:
            self.counter += 1
        stop = self.counter > self.patience
        return stop
    
    def load_best_parameters(self, model):
        best_state = torch.load(self.exp_dir / "best_params.pt")
        model.load_state_dict(best_state)

def optim_aligner(aligner, classifier, optimizer, patch_arbors, norm_scale, norm_loss_frac, perturb):
    optimizer.zero_grad()
    aligner_output = aligner(patch_arbors)
    aligned_arbors = (patch_arbors + aligner_output if perturb else aligner_output)
    norm_loss = torch.linalg.norm(aligned_arbors - patch_arbors, dim = 1).mean()
    classifier_loss = classifier.get_loss(aligned_arbors)
    loss = norm_loss_frac*norm_scale*norm_loss + (1 - norm_loss_frac)*classifier_loss
    loss.backward()
    optimizer.step()
    return (norm_loss, classifier_loss)

def optim_classifier(aligner, classifier, optimizer, patch_arbors, em_arbors, perturb):
    (num_em, num_patch) = (em_arbors.shape[0], patch_arbors.shape[0])
    optimizer.zero_grad()
    aligner_output = aligner(patch_arbors)
    aligned_arbors = (patch_arbors + aligner_output if perturb else aligner_output)
    labels = torch.cat([torch.zeros(num_em), torch.ones(num_patch)]).long().to(patch_arbors.device)
    log_odds = classifier(torch.cat([em_arbors, aligned_arbors]))[:, 0]
    loss = torch.nn.functional.cross_entropy(torch.stack([-log_odds, log_odds], 1), labels)
    loss.backward()
    optimizer.step()
    correct = (log_odds[:num_em] < 0).int().sum() + (log_odds[num_em:] > 0).int().sum()
    return (aligned_arbors, correct)

def run_validation(aligner, classifier, patch_val, em_val, perturb):
    aligner_output = aligner(patch_val)
    aligned_arbors = (patch_val + aligner_output if perturb else aligner_output)
    correct = (classifier(aligned_arbors) > 0).int().sum() + (classifier(em_val) < 0).int().sum()
    norm_loss = torch.linalg.norm(aligned_arbors - patch_val, dim = 1).mean()
    class_loss = classifier.get_loss(aligned_arbors)
    return (norm_loss, class_loss, correct)

def fit_validators(loader, gauss_noise, aligner = None, perturb = False, val_count = 100):
    patch_arbors = loader.dataset.patch_arbors
    em_arbors = loader.dataset.em_arbors[:len(patch_arbors)]
    if aligner is not None:
        aligner_output = aligner(patch_arbors)
        patch_arbors = (patch_arbors + aligner_output if perturb else aligner_output)
    X_train = torch.cat([em_arbors[val_count:], patch_arbors[val_count:]]).detach().cpu().numpy()
    X_test = torch.cat([em_arbors[:val_count], patch_arbors[:val_count]]).detach().cpu().numpy()
    X_train = X_train + gauss_noise*np.random.normal(0, 1, X_train.shape)
    X_test = X_test + gauss_noise*np.random.normal(0, 1, X_test.shape)
    y_train = np.concatenate([-np.ones([em_arbors.shape[0] - val_count]), np.ones([patch_arbors.shape[0] - val_count])])
    y_test = np.concatenate([-np.ones([val_count]), np.ones([val_count])])
    ridge_acc = RidgeClassifier(alpha = 5).fit(X_train, y_train).score(X_test, y_test)
    forest_acc = 1 #RandomForestClassifier().fit(X_train, y_train).score(X_test, y_test)
    mlp_acc = MLPClassifier((300, 200, 100)).fit(X_train, y_train).score(X_test, y_test)
    return (ridge_acc, forest_acc, mlp_acc)

def print_results(results, epoch):
    (norm, entropy) = results["metrics"][:2] / results["count"][0]
    acc = results["metrics"][2] / results["count"].sum()
    (val_norm, val_entropy) = results["val_metrics"][:2] / results["val_count"][0]
    val_acc = results["val_metrics"][2] / results["val_count"].sum()
    print(f"Epoch {epoch} ----------------------------------------------")
    print(f"Training Perturbation Norm: {norm:.4f} | Training Classification Loss: {entropy:.4f} | Training Accuracy: {100*acc:.2f}%")
    print(f"Validation Perturbation Norm: {val_norm:.4f} | Validation Classification Loss: {val_entropy:.4f} | Validation Accuracy: {100*val_acc:.2f}%")
    if "validators" in results:
        (ridge_acc, forest_acc, mlp_acc) = results["validators"]
        print(f"Ridge: {100*ridge_acc:.2f}% | Random Forest: {100*forest_acc:.2f}% | Neural Network: {100*mlp_acc:.2f}%")

def log_tensorboard(writer, results, epoch):
    (norm, entropy) = results["metrics"][:2] / results["count"][0]
    acc = results["metrics"][2] / results["count"].sum()
    (val_norm, val_entropy) = results["val_metrics"][:2] / results["val_count"][0]
    val_acc = results["val_metrics"][2] / results["val_count"].sum()
    writer.add_scalars("norm", {"train": norm, "val": val_norm}, epoch)
    writer.add_scalars("class loss", {"train": entropy, "val": val_entropy}, epoch)
    writer.add_scalars("acc", {"train": acc, "val": val_acc}, epoch)
    if "validators" in results:
        (ridge_acc, forest_acc, mlp_acc) = results["validators"]
        writer.add_scalars("validators", {"ridge": ridge_acc, "forest": forest_acc, "mlp": mlp_acc}, epoch)

def write_baseline(writer, loader, gauss_noise, aligner = None, perturb = False, val_count = 100):
    (ridge_acc, forest_acc, mlp_acc) = fit_validators(loader, gauss_noise, aligner, perturb, val_count)
    writer.add_scalars("validator baseline", {"ridge": ridge_acc, "forest": forest_acc, "mlp": mlp_acc})

def train(aligner, classifier, train_loader, val_loader, params):
    writer = SummaryWriter(params["dir"], flush_secs = 1)
    # stopper = EarlyStopping(params["dir"], params["patience"], params["delta"])
    norm_scale = 1
    aligner_optimizer = torch.optim.Adam(aligner.parameters(), lr = params["aligner_lr"])
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr = params["classifier_lr"])
    write_baseline(writer, train_loader, params["gauss_noise"])
    for epoch in range(params["epochs"]):
        print(f"Epoch: {epoch + 1}", end = "\r")
        epoch_results = {"count": np.zeros([2]), "metrics": np.zeros([3]), "val_count": np.zeros([2]), "val_metrics": np.zeros([3])}
        for arbors in iter(train_loader):
            (em_arbors, patch_arbors) = (arbors[:, 0], arbors[:, 1])
            (norm_loss, classifier_loss) = optim_aligner(aligner, classifier, aligner_optimizer, patch_arbors, norm_scale, params["norm_loss_frac"], params["perturb"])
            (aligned_arbors, correct) = optim_classifier(aligner, classifier, classifier_optimizer, patch_arbors, em_arbors, params["perturb"])
            with torch.no_grad():
                norm_scale = (classifier_loss / norm_loss if (type(norm_scale) == int) else norm_scale)
            epoch_results["metrics"] += np.asarray([norm_loss.item()*patch_arbors.shape[0], classifier_loss.item()*patch_arbors.shape[0], correct.item()])
            epoch_results["count"] += np.asarray([patch_arbors.shape[0], em_arbors.shape[0]])
        with torch.no_grad():
            for val_arbors in iter(val_loader):
                (em_val_arbors, patch_val_arbors) = (val_arbors[:, 0], val_arbors[:, 1])
                (val_norm_loss, val_classifier_loss, val_correct) = run_validation(aligner, classifier, patch_val_arbors, em_val_arbors, params["perturb"])
                epoch_results["val_metrics"] += np.asarray([val_norm_loss.item()*patch_val_arbors.shape[0], val_classifier_loss.item()*patch_val_arbors.shape[0], val_correct.item()])
                epoch_results["val_count"] += np.asarray([patch_val_arbors.shape[0], em_val_arbors.shape[0]])
            if epoch % params["validator_step"] == 0 or epoch + 1 == params["epochs"]:
                epoch_results["validators"] = fit_validators(train_loader, params["gauss_noise"], aligner, params["perturb"])
        # if stopper.stop_check(epoch_results, aligner, classifier, epoch) or (epoch + 1 == params["epochs"]):
        #     stopper.load_best_parameters(aligner, classifier)
        log_tensorboard(writer, epoch_results, epoch + 1)
    writer.flush()
    writer.close()
    print("")

device = "cuda"
activation = "softplus"
aligner_hidden_dims = (100, 10, 100)
classifier_hidden_dims = (100,)
batch_size = 1024
oversample = True

params = dict(
    dir = "data/alignment/logs/oversample",
    perturb = False,
    gauss_noise = 0.0,
    epochs = 5000,
    aligner_lr = 1e-4,
    classifier_lr = 1e-5,
    norm_loss_frac = 0.9,
    validator_step = 10)

(em_arbors, patch_arbors) = load_raw_data()
(X_em_train, X_em_test) = train_test_split(em_arbors, test_size = 0.25)
(X_patch_train, X_patch_test) = train_test_split(patch_arbors, test_size = 0.25)
if not oversample:
    X_em_train = X_em_train[:len(X_patch_train)]
    X_em_test = X_em_test[:len(X_patch_test)]

train_dataset = ArborDataset(X_em_train, X_patch_train, device)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)
val_dataset = ArborDataset(X_em_test, X_patch_test, device)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size)

aligner = Aligner(classifier_hidden_dims, activation)
classifier = Classifier(classifier_hidden_dims, params["gauss_noise"])

train(aligner.float().to(device), classifier.float().to(device), train_dataloader, val_dataloader, params)
