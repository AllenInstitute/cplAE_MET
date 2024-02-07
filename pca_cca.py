import yaml
import pathlib
import argparse
from copy import deepcopy
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from scipy.linalg import sqrtm

from data import MET_Data, MET_Dataset

def filter_specimens(met_data, specimen_ids, config):
    platforms = config["select"]["platforms"]
    specimens = met_data.query(specimen_ids, platforms = platforms)["specimen_id"]
    return specimens

class CCA_extended(CCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, data_dict):
        data_list = list(data_dict.items())
        data_list.sort(key = lambda modal_X: modal_X[0])
        data_arrays = [X for (modal, X) in data_list]
        self.var_map = {data_list[0][0]: "x", data_list[1][0]: "y"}
        super().fit(*data_arrays)

    def transform(self, X, in_modal):
        var = self.var_map[in_modal]
        x = X - getattr(self, f"_{var}_mean")
        x = x / getattr(self, f"_{var}_std")
        score = np.dot(x, getattr(self, f"{var}_rotations_"))
        return score

    def inverse_transform(self, Z, out_modal):
        var = self.var_map[out_modal]
        x = np.matmul(Z, getattr(self, f"{var}_loadings_").T)
        x = x * getattr(self, f"_{var}_std")
        x = x + getattr(self, f"_{var}_mean")
        return x

class PCA_CCA():
    def __init__(self, config):
        self.pca_dims = {modal: config[f"pca_dim_{modal}"] for modal in config["modalities"]}
        self.cca_dim = config["latent_dim"]

    def fit(self, data_dict):
        self.pca_dims = {modal: (dim if dim > 0 else data_dict[modal].shape[1]) for (modal, dim) in self.pca_dims.items()}
        self.pca = {modal: PCA(self.pca_dims[modal]).fit(X) for (modal, X) in data_dict.items()}
        pca_data = {modal: self.pca[modal].transform(X) for (modal, X) in data_dict.items()}
        self.cca = CCA_extended(n_components = self.cca_dim, scale = True, max_iter = 10000, tol = 1e-06, copy = True)
        self.cca.fit(pca_data)

    def __call__(self, X, in_modal, out_modals):
        X = X.reshape([X.shape[0], -1])
        latent = self.get_latent(X, in_modal)
        outputs = [self.get_reconstruction(latent, modal) for modal in out_modals]
        return (latent, outputs)

    def get_latent(self, X, in_modal):
        X = X.reshape([X.shape[0], -1])
        X_pca = self.pca[in_modal].transform(X)
        z = self.cca.transform(X_pca, in_modal)
        return z
    
    def get_reconstruction(self, z, out_modal):
        x_cca = self.cca.inverse_transform(z, out_modal)
        x = self.pca[out_modal].inverse_transform(x_cca)
        return x
    
    def save(self, exp_dir):
        for (modal, pca) in self.pca.items():
            with open(exp_dir / f"{modal}_pca.pkl", "wb") as target:
                pickle.dump(pca, target)
        with open(exp_dir / "cca.pkl", "wb") as target:
            pickle.dump(self.cca, target)
    
    def load(self, exp_dir):
        self.pca = {}
        for path in exp_dir.glob("*_pca.pkl"):
            modal = path.name.split("_")[0]
            with open(path, "rb") as target:
                pca = pickle.load(target)
            self.pca[modal] = pca
        with open(exp_dir / "cca.pkl", "rb") as target:
            self.cca = pickle.load(target)

def pc_cca(XT, XE, train_ind, pc_dim_T, pc_dim_E, cca_dim):
    """Reduce dimensionality of XT and XE with PCA, and then obtained an co-ordinated representations with CCA.

    Args:
        XT: numpy arrays cells x features
        XE: numpy arrays cells x features
        train_ind: Principle components and canonical components are obtained with this set of cells. 
        pc_dim_T (int): Numper of principle components for XT
        pc_dim_E (int): Numper of principle components for XE
        cca_dim (int): shared space dimensionality

    Returns:
        zT_white: zT is centered and transformed to have unit diagonal covariance (whitening transformation)
        zE_white: zE is centered and transformed to have unit diagonal covariance (whitening transformation)
        XrT
        XrE
        XrT_from_XE
        XrE_from_XT
    """

    XT = deepcopy(XT)
    XE = deepcopy(XE)

    #Reduce dims of T data
    if pc_dim_T is not None:
        pcaT = PCA(n_components=pc_dim_T)
        pcaT.fit_transform(XT[train_ind, :])
        XTpc = pcaT.transform(XT)
    else:
        XTpc = XT

    #Reduce dims of E data
    if pc_dim_E is not None:
        pcaE = PCA(n_components=pc_dim_E)
        pcaE.fit_transform(XE[train_ind, :])
        XEpc = pcaE.transform(XE)
    else:
        XEpc = XE

    #CCA on T and E data
    cca = CCA_extended(n_components=cca_dim, scale=True, max_iter=10000, tol=1e-06, copy=True)
    cca.fit(XTpc[train_ind, :], XEpc[train_ind, :])
    zT, zE = cca.transform(XTpc, XEpc)

    zT_white = zT - np.mean(zT, axis=0)
    zT_white = np.matmul(zT_white, sqrtm(
        np.linalg.inv(np.cov(np.transpose(zT_white)))))

    zE_white = zE - np.mean(zE, axis=0)
    zE_white = np.matmul(zE_white, sqrtm(
        np.linalg.inv(np.cov(np.transpose(zE_white)))))

    #Within modality reconstruction
    XrTpc, XrEpc = cca.inverse_transform_xy(zT, zE)
    XrT = pcaT.inverse_transform(XrTpc)
    XrE = pcaE.inverse_transform(XrEpc)

    #Cross modality reconstruction
    XrTpc_from_zE, XrEpc_from_zT = cca.inverse_transform_xy(zE, zT)
    XrT_from_XE = pcaT.inverse_transform(XrTpc_from_zE)
    XrE_from_XT = pcaE.inverse_transform(XrEpc_from_zT)

    return zT_white, zE_white, XrT, XrE, XrT_from_XE, XrE_from_XT

def train_pca(exp_dir, config, train_dataset):
    pca_cca = PCA_CCA(config)
    data_dicts = [batch[0] for batch in train_dataset]
    data_arrays = {modal: np.concatenate([X[modal][0] for X in data_dicts]) for modal in config["modalities"]}
    data_flat = {modal: X.reshape([X.shape[0], -1]) for (modal, X) in data_arrays.items()}
    pca_cca.fit(data_flat)
    pca_cca.save(exp_dir)

def train_model(config, exp_dir):
    met_data = MET_Data(config["data_file"])
    num_folds = config["folds"]
    if num_folds > 0:
        indices = list(met_data.get_stratified_KFold(config["folds"], seed = config["seed"]))
    else:
        (train_ids, test_ids) = met_data.get_stratified_split(config["val_split"], seed = config["seed"])
        indices = [(train_ids, test_ids)]
        num_folds = 1
    for (fold, (train_ids, test_ids)) in enumerate(indices, 1):
        print(f"Processing fold {fold} / {len(indices)}.")
        exp_fold_dir = exp_dir / f"fold_{fold}"
        exp_fold_dir.mkdir(exist_ok = True)
        filtered_train_ids = filter_specimens(met_data, train_ids, config)
        train_dataset = MET_Dataset(met_data, 1024, config["modal_frac"], filtered_train_ids)
        np.savez_compressed(exp_fold_dir / "train_test_ids.npz", **{"train": train_ids, "test": test_ids})
        train_pca(exp_fold_dir, config, train_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help = "Name of experiment.")
    parser.add_argument("config_path", help = "path to config yaml file")
    args = parser.parse_args()
    with open(args.config_path, "r") as target:
        config = yaml.safe_load(target)
    exp_dir = pathlib.Path(config["output_dir"]) / args.exp_name
    exp_dir.mkdir(exist_ok = True)
    train_model(config, exp_dir)