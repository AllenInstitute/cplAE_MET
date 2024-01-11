import contextlib

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
    return (arbors, anno_unnorm, anno_samp)

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
    def __init__(self, hidden_dims):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.layer_names = []
        all_dims = [960] + list(hidden_dims) + [960]
        for (i, (in_dim, out_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:]), 1):
            setattr(self, f"l{i}", torch.nn.Linear(in_dim, out_dim))
            self.layer_names.append(f"l{i}")
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        for layer_name in self.layer_names[:-1]:
            layer = getattr(self, layer_name)
            X = self.relu(layer(X))
        final_layer = getattr(self, self.layer_names[-1])
        output = torch.nn.functional.softplus(final_layer(X))
        return output
    
class Classifier(torch.nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        self.hidden_dims = hidden_dims
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
    
def optim_aligner(aligner, classifier, optimizer, patch_arbors, norm_scale, norm_loss_frac):
    optimizer.zero_grad()
    perturbation = aligner(patch_arbors)
    aligned_arbors = perturbation #patch_arbors + perturbation
    norm_loss = torch.linalg.norm(perturbation - patch_arbors, dim = 1).mean() #torch.linalg.norm(perturbation, dim = 1).mean()
    classifier_loss = classifier.get_loss(aligned_arbors)
    loss = norm_loss_frac*norm_scale*norm_loss + (1 - norm_loss_frac)*classifier_loss
    loss.backward()
    optimizer.step()
    return (norm_loss, classifier_loss)

def optim_classifier(aligner, classifier, optimizer, patch_arbors, em_arbors):
    (num_em, num_patch) = (em_arbors.shape[0], patch_arbors.shape[0])
    optimizer.zero_grad()
    perturbation = aligner(patch_arbors)
    aligned_arbors = perturbation #patch_arbors + perturbation
    labels = torch.cat([torch.zeros(num_em), torch.ones(num_patch)]).long().to(patch_arbors.device)
    log_odds = classifier(torch.cat([em_arbors, aligned_arbors]))[:, 0]
    loss = torch.nn.functional.cross_entropy(torch.stack([-log_odds, log_odds], 1), labels)
    loss.backward()
    optimizer.step()
    correct = (log_odds[:num_em] < 0).int().sum() + (log_odds[num_em:] > 0).int().sum()
    return (aligned_arbors, correct)

def run_validation(aligner, classifier, patch_val, em_val):
    perturbation = aligner(patch_val)
    aligned = patch_val + perturbation
    correct = (classifier(patch_val) > 0).int().sum() + (classifier(em_val) > 0).int().sum()
    norm_loss = torch.linalg.norm(perturbation, dim = 1).mean()
    class_loss = classifier.get_loss(patch_val)
    return (norm_loss, class_loss, correct)

def fit_validators(aligned_arbors, em_arbors, val_count = 100):
    X_train = torch.cat([em_arbors[val_count:], aligned_arbors[val_count:]]).detach().cpu().numpy()
    X_test = torch.cat([em_arbors[:val_count], aligned_arbors[:val_count]]).detach().cpu().numpy()
    y_train = np.concatenate([-np.ones([em_arbors.shape[0] - 100]), np.ones([aligned_arbors.shape[0] - 100])])
    y_test = np.concatenate([-np.ones([100]), np.ones([100])])
    ridge_acc = RidgeClassifier(alpha = 5).fit(X_train, y_train).score(X_test, y_test)
    forest_acc = RandomForestClassifier().fit(X_train, y_train).score(X_test, y_test)
    mlp_acc = MLPClassifier(((300, 200, 100))).fit(X_train, y_train).score(X_test, y_test)
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

def train(aligner, classifier, train_loader, val_loader, em_arbors, em_val_arbors, epochs, aligner_lr, classifier_lr, norm_loss_frac):
    norm_scale = 1
    aligner_optimizer = torch.optim.Adam(aligner.parameters(), lr = aligner_lr)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr = classifier_lr)
    all_results = []
    for epoch in range(epochs):
        epoch_results = {"count": np.zeros([2]), "metrics": np.zeros([3]), "val_count": np.zeros([2]), "val_metrics": np.zeros([3])}
        for (step, (patch_arbors, specimen_ids)) in enumerate(iter(train_loader), 1):
            (norm_loss, classifier_loss) = optim_aligner(aligner, classifier, aligner_optimizer, 
                                                                       patch_arbors, norm_scale, norm_loss_frac)
            (aligned_arbors, correct) = optim_classifier(aligner, classifier, classifier_optimizer, patch_arbors, em_arbors)
            with torch.no_grad():
                norm_scale = (classifier_loss / norm_loss if (type(norm_scale) == int) else norm_scale)
            epoch_results["metrics"] += np.asarray([norm_loss.item()*patch_arbors.shape[0], classifier_loss.item()*patch_arbors.shape[0], correct.item()])
            epoch_results["count"] += np.asarray([patch_arbors.shape[0], em_arbors.shape[0]])
        with torch.no_grad():
            for (val_arbors, _) in iter(val_loader):
                (val_norm_loss, val_classifier_loss, val_correct) = run_validation(aligner, classifier, val_arbors, em_val_arbors)
                epoch_results["val_metrics"] += np.asarray([val_norm_loss.item()*val_arbors.shape[0], val_classifier_loss.item()*val_arbors.shape[0], val_correct.item()])
                epoch_results["val_count"] += np.asarray([val_arbors.shape[0], em_val_arbors.shape[0]])
            if epoch % 1000 == 0 or epoch + 1 == epochs:
                epoch_results["validators"] = fit_validators(aligned_arbors, em_arbors)
        all_results.append(epoch_results)
        print_results(epoch_results, epoch + 1)
    return all_results

device = "mps"
num_epochs = 30000
batch_size = 2048
aligner_hidden_dims = (960, 960, 960, 960, 960) #(100, 10, 100)
classifier_hidden_dims = (100,)
aligner_lr = 1e-4
classifier_lr = 1e-5
norm_loss_frac = 1

(arbors, anno_unnorm, anno_samp) = load_raw_data()
anno_exc = anno_samp.query("`class` == 'exc'")
X_exc = select(arbors, anno_exc)[..., 2:].reshape((-1, 960))
y_exc = np.where(anno_exc["platform"] == "EM", 0, 1)

(X_train, X_test, y_train, y_test) = train_test_split(X_exc, y_exc)
(X_patch_test, y_patch_test) = (X_test[y_test == 1], y_test[y_test == 1])
(X_em_train, X_em_test) = (X_train[y_train == 0], X_test[y_test == 0])

train_dataset = ExcDataset(X_train[y_train == 1], y_train[y_train == 1], device)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_dataset = ExcDataset(X_test[y_test == 1], y_test[y_test == 1], device)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 1024)

aligner = Aligner(classifier_hidden_dims)
classifier = Classifier(classifier_hidden_dims)

results = train(aligner.float().to(device), classifier.float().to(device), train_dataloader, val_dataloader, 
                torch.from_numpy(X_em_train).float().to(device), torch.from_numpy(X_em_test).float().to(device), 
                num_epochs, aligner_lr, classifier_lr, norm_loss_frac)

accs = [(epoch, result_dict["validators"]) for (epoch, result_dict) in enumerate(results, 1) if "validators" in result_dict]
epochs = [epoch for (epoch, _) in accs]
(ridge_accs, forest_accs, mlp_accs) = ([tupl[1][0] for tupl in accs], [tupl[1][1] for tupl in accs], [tupl[1][2] for tupl in accs])
plt.plot(epochs, ridge_accs, label = "Ridge")
plt.plot(epochs, forest_accs, label = "Random Forest")
plt.plot(epochs, mlp_accs, label = "Neural Network")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("EM + Patch-seq Accuracy")
plt.show()