import os
import sys
import re
import umap
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
import cplAE_MET.utils.utils as ut
from sklearn.preprocessing import StandardScaler
from cplAE_MET.utils.utils import loadpkl
from cplAE_MET.models.optuna_utils import Leiden_community_detection
from cplAE_MET.models.classification_functions import run_LDA
from sklearn.model_selection import KFold





def get_TE_pkl_keys(parent_dir_path, listdir):
    '''
    Get list of pkl files and return the hyperparameters used in the model as well as the model output
    Args:
        parent_dir_path: path to the parent folder which contains folders with multiple pkl files in them
        listdir: list of all the folders that have multiple pkl files and we want to analyze
    '''
    output = {}
    df = pd.DataFrame()
    for f in listdir:
        if f.endswith(".pkl"):
            alphaT = re.search('aT_(.*)_aE', f).group(1) if re.search('aT_(.*)_aE', f) else np.nan
            alphaE = re.search('aE_(.*)_lambda_TE', f).group(1) if re.search('aE_(.*)_lambda_TE', f) else np.nan
            lambda_TE = re.search('lambda_TE_(.*)_lambda_tune_TE', f).group(1) if re.search('lambda_TE_(.*)_lambda_tune_TE', f) else np.nan
            lambda_tune_TE = re.search('lambda_tune_TE_(.*)_Enoise', f).group(1) if re.search('lambda_tune_TE_(.*)_Enoise', f) else np.nan
            aug_dec = re.search('aug_dec_(.*)_ld', f).group(1) if re.search('aug_dec_(.*)_ld', f) else 0
            latent_dim = re.search('ld_(.*)_ne', f).group(1) if re.search('ld_(.*)_ne', f) else np.nan
            fold = re.search('fold_(.*).pkl', f).group(1) if re.search('fold_(.*).pkl', f) else np.nan
            dir = parent_dir_path + f
            output[(alphaT, alphaE, lambda_TE, lambda_tune_TE, latent_dim, aug_dec, fold)] = ut.loadpkl(dir)
            classification_acc_zt = output[(alphaT, alphaE, lambda_TE, lambda_tune_TE, latent_dim, aug_dec, fold)]['classification_acc_zt']
            classification_acc_ze = output[(alphaT, alphaE, lambda_TE, lambda_tune_TE, latent_dim, aug_dec, fold)]['classification_acc_ze']
            recon_loss_xt = output[(alphaT, alphaE, lambda_TE, lambda_tune_TE, latent_dim, aug_dec, fold)]['recon_loss_xt']
            recon_loss_xe = output[(alphaT, alphaE, lambda_TE, lambda_tune_TE, latent_dim, aug_dec, fold)]['recon_loss_xe']


            df1 = pd.DataFrame({'alphaT': [float(alphaT.replace("-", "."))],
            "alphaE": [float(alphaE.replace("-", "."))],
            "lambda_TE": [float(lambda_TE.replace("-", "."))],
            "lambda_tune_TE": [float(lambda_tune_TE.replace("-", "."))],
            "aug_dec": [float(aug_dec)],
            "latent_dim": [int(latent_dim)],
            "fold": [int(fold)],
            "classification_acc_zt": [classification_acc_zt],
            "classification_acc_ze": [classification_acc_ze],
            "recon_loss_xt": [float(recon_loss_xt)],
            "recon_loss_xe": [float(recon_loss_xe)]})

            df = pd.concat([df, df1], ignore_index=True)

    return output, df


def get_T_ME_version_0_0_pkl_keys(parent_dir_path, listdir):
    '''
    Get list of pkl files and return the hyperparameters used in the model as well as the model output
    Args:
        parent_dir_path: path to the parent folder which contains folders with multiple pkl files in them
        listdir: list of all the folders that have multiple pkl files and we want to analyze
    '''
    output = {}
    df = pd.DataFrame()
    for f in listdir:
        if f.endswith(".pkl"):
            aT = re.search('aT_(.*)_aM_', f).group(1) if re.search('aT_(.*)_aM_', f) else np.nan
            cond1 = re.search('aM_(.*)_asd_', f)
            cond2 = re.search('aM_(.*)_aE_', f)
            aM = re.search('aM_(.*)_asd_', f).group(1) if cond1 else re.search('aM_(.*)_aE_', f).group(
                1) if cond2 else np.nan
            asd = re.search('asd_(.*)_aE_', f).group(1) if re.search('asd_(.*)_aE_', f) else aM
            aE = re.search('aE_(.*)_aME_', f).group(1) if re.search('aE_(.*)_aME_', f) else np.nan
            aME = re.search('aME_(.*)_lambda_ME_T', f).group(1) if re.search('aME_(.*)_lambda_ME_T', f) else np.nan
            lambda_ME_T = re.search('lambda_ME_T_(.*)_lambda_tune', f).group(1) if re.search(
                'lambda_ME_T_(.*)_lambda_tune', f) else np.nan
            lambda_ME_M = re.search('lambda_ME_M_(.*)_lambda', f).group(1) if re.search('lambda_ME_M_(.*)_lambda',
                                                                                        f) else np.nan
            lambda_tune_ME_T = re.search('lambda_tune_ME_T_(.*)_lambda_ME_M', f).group(1) if re.search(
                'lambda_tune_ME_T_(.*)_lambda_ME_M', f) else np.nan
            aug_dec = re.search('aug_dec_(.*)_Enoise', f).group(1) if re.search('aug_dec_(.*)_Enoise', f) else np.nan
            cond1 = re.search('lambda_ME_E_(.*)_aug_dec', f)
            cond2 = re.search('lambda_ME_E_(.*)_Enoise', f)
            lambda_ME_E = re.search('lambda_ME_E_(.*)_aug_dec', f).group(1) if cond1 else re.search(
                'lambda_ME_E_(.*)_Enoise', f).group(1) if cond2 else np.nan
            latent_dim = re.search('ld_(.*)_ne', f).group(1) if re.search('ld_(.*)_ne', f) else np.nan
            fold = re.search('fold_(.*).pkl', f).group(1) if re.search('fold_(.*).pkl', f) else np.nan

            dir = parent_dir_path + f

            output[(aT, aE, aM, asd, aME, lambda_ME_T, lambda_tune_ME_T, lambda_ME_M, lambda_ME_E, latent_dim, aug_dec,
                    fold)] = ut.loadpkl(dir)
            classification_acc_zt = output[(
            aT, aE, aM, asd, aME, lambda_ME_T, lambda_tune_ME_T, lambda_ME_M, lambda_ME_E, latent_dim, aug_dec, fold)][
                'classification_acc_zt']
            classification_acc_ze = output[(
            aT, aE, aM, asd, aME, lambda_ME_T, lambda_tune_ME_T, lambda_ME_M, lambda_ME_E, latent_dim, aug_dec, fold)][
                'classification_acc_ze']
            classification_acc_zme = output[(
            aT, aE, aM, asd, aME, lambda_ME_T, lambda_tune_ME_T, lambda_ME_M, lambda_ME_E, latent_dim, aug_dec, fold)][
                'classification_acc_zme']
            classification_acc_zm = output[(
            aT, aE, aM, asd, aME, lambda_ME_T, lambda_tune_ME_T, lambda_ME_M, lambda_ME_E, latent_dim, aug_dec, fold)][
                'classification_acc_zm']

            df1 = pd.DataFrame({
                'aT': [float(aT.replace("-", "."))],
                "aE": [float(aE.replace("-", "."))],
                "aM": [float(aM.replace("-", "."))],
                "asd": [float(asd.replace("-", "."))],
                "aME": [float(aME.replace("-", "."))],
                "lambda_ME_T": [float(lambda_ME_T.replace("-", "."))],
                "lambda_ME_M": [float(lambda_ME_M.replace("-", "."))],
                "lambda_ME_E": [float(lambda_ME_E.replace("-", "."))],
                "lambda_tune_ME_T": [float(lambda_tune_ME_T.replace("-", "."))],
                "latent_dim": [int(latent_dim)],
                "aug_dec": [float(aug_dec)],
                "fold": [int(fold)],
                "classification_acc_zt": [float(classification_acc_zt)],
                "classification_acc_ze": [float(classification_acc_ze)],
                "classification_acc_zm": [float(classification_acc_zm)],
                "classification_acc_zme": [float(classification_acc_zme)]})

            df = pd.concat([df, df1], ignore_index=True)

    return output, df

def get_T_ME_version_1_0_pkl_keys(parent_dir_path, listdir):
    '''
    Get list of pkl files and return the hyperparameters used in the model as well as the model output
    Args:
        parent_dir_path: path to the parent folder which contains folders with multiple pkl files in them
        listdir: list of all the folders that have multiple pkl files and we want to analyze
    '''
    output = {}
    df = pd.DataFrame()
    for f in listdir:
        if f.endswith(".pkl"):
            aT = re.search('aT_(.*)_aM_', f).group(1) if re.search('aT_(.*)_aM_', f) else np.nan
            cond1 = re.search('aM_(.*)_asd_', f)
            cond2 = re.search('aM_(.*)_aE_', f)
            aM = re.search('aM_(.*)_asd_', f).group(1) if cond1 else re.search('aM_(.*)_aE_', f).group(
                1) if cond2 else np.nan
            asd = re.search('asd_(.*)_aE_', f).group(1) if re.search('asd_(.*)_aE_', f) else aM
            aE = re.search('aE_(.*)_aME_', f).group(1) if re.search('aE_(.*)_aME_', f) else np.nan
            aME = re.search('aME_(.*)_lambda_ME_T', f).group(1) if re.search('aME_(.*)_lambda_ME_T', f) else np.nan
            lambda_ME_T = re.search('lambda_ME_T_(.*)_lambda_E_T', f).group(1) if re.search(
                'lambda_ME_T_(.*)_lambda_E_T', f) else np.nan
            lambda_E_T = re.search('lambda_E_T_(.*)_lambda_M_T', f).group(1) if re.search('lambda_E_T_(.*)_lambda_M_T',
                                                                                        f) else np.nan
            lambda_M_T = re.search('lambda_M_T_(.*)_lambda_tune_ME_T', f).group(1) if re.search(
                'lambda_M_T_(.*)_lambda_tune_ME_T', f) else np.nan
            aug_dec = re.search('aug_dec_(.*)_Enoise', f).group(1) if re.search('aug_dec_(.*)_Enoise', f) else np.nan
            lambda_tune_ME_T = re.search('lambda_tune_ME_T_(.*)_aug', f).group(1) if re.search(
                'lambda_tune_ME_T_(.*)_aug', f) else np.nan
            latent_dim = re.search('ld_(.*)_ne', f).group(1) if re.search('ld_(.*)_ne', f) else np.nan
            fold = re.search('fold_(.*).pkl', f).group(1) if re.search('fold_(.*).pkl', f) else np.nan
            dir = parent_dir_path + f

            output[(aT, aE, aM, asd, aME, lambda_ME_T, lambda_tune_ME_T, lambda_E_T, lambda_M_T, latent_dim, aug_dec,
                    fold)] = ut.loadpkl(dir)
            classification_acc_zt = output[(
            aT, aE, aM, asd, aME, lambda_ME_T, lambda_tune_ME_T, lambda_E_T, lambda_M_T, latent_dim, aug_dec, fold)][
                'classification_acc_zt']
            classification_acc_ze = output[(
            aT, aE, aM, asd, aME, lambda_ME_T, lambda_tune_ME_T, lambda_E_T, lambda_M_T, latent_dim, aug_dec, fold)][
                'classification_acc_ze']
            classification_acc_zme = output[(
            aT, aE, aM, asd, aME, lambda_ME_T, lambda_tune_ME_T, lambda_E_T, lambda_M_T, latent_dim, aug_dec, fold)][
                'classification_acc_zme']
            classification_acc_zm = output[(
            aT, aE, aM, asd, aME, lambda_ME_T, lambda_tune_ME_T, lambda_E_T, lambda_M_T, latent_dim, aug_dec, fold)][
                'classification_acc_zm']

            df1 = pd.DataFrame({
                'aT': [float(aT.replace("-", "."))],
                "aE": [float(aE.replace("-", "."))],
                "aM": [float(aM.replace("-", "."))],
                "asd": [float(asd.replace("-", "."))],
                "aME": [float(aME.replace("-", "."))],
                "lambda_ME_T": [float(lambda_ME_T.replace("-", "."))],
                "lambda_E_T": [float(lambda_E_T.replace("-", "."))],
                "lambda_M_T": [float(lambda_M_T.replace("-", "."))],
                "lambda_tune_ME_T": [float(lambda_tune_ME_T.replace("-", "."))],
                "latent_dim": [int(latent_dim)],
                "aug_dec": [float(aug_dec)],
                "fold": [int(fold)],
                "classification_acc_zt": [float(classification_acc_zt)],
                "classification_acc_ze": [float(classification_acc_ze)],
                "classification_acc_zm": [float(classification_acc_zm)],
                "classification_acc_zme": [float(classification_acc_zme)]})

            df = pd.concat([df, df1], ignore_index=True)

    return output, df

def compute_r2score(X, Xr, which_cols, which_rows=None):
    """
    Compute Rsqueard between the two datasets for the given columns using the given rows

    Args:
        X: input data
        Xr: reconstructed data
        which_cols: a list of columns that we need to compute the r2_score for
        which_rows: only these rows in the given cols will be used, if none then all the rows will be used

    return:
        r2_score: a list of r2_scores for the given columns of the two datasets using the given rows
    """
    if which_rows is not None:
        X = X[which_rows, :]
        Xr = Xr[which_rows, :]
    return [r2_score(X[:, i][~np.isnan(X[:, i])], Xr[:, i][~np.isnan(X[:, i])]) for i in which_cols]


def check_for_nan(nparray):
    """Takes a nparray and check if there is any nan values in it.

    Args:
    nparray: a numpy array

    Returns:
    True or False
    """
    return np.sum(np.all(np.isnan(nparray), axis=1)) > 0


def return_nan_index_col(x):
    """Takes an array and returns the index and col of all nan values

    Args:
    x (np.array): Array

    Returns:
    rows, cols
    """
    rows, cols = np.where(np.isnan(x))
    return rows, cols


def drop_nan_rows_or_cols(nparray, axis):
    """Takes a nparray and drop rows or cols (specified by axis) with nan values

    Args:
    nparray: a numpy array
    axis: drop rows (axis=0) or drop columns (axis=1)

    Returns:
    final_nparray: a numpy array
    dropped: a list of dropped row or column indices
    """
    mask = np.all(np.isnan(nparray), axis=axis)
    final_nparray = nparray[~mask]
    dropped = np.where(mask)[0]
    return final_nparray, dropped


def get_PCA_explained_variance_ratio_at_thr(nparray, threshold, show_plots=True):
    """Apply PCA on a given numpy array, compute the cumulative explained variance ratio as
    a function of number of components, plot it and return the number of components needed
    at the given threshold

    Args:
    nparray: a numpy array
    threshold: The value for which the explained variance ratio is requested

    Returns:
    n_components : returns the explained variance ratio
    """
    pca = PCA()
    pca_feature = pca.fit_transform(nparray)
    pca_expl_var = pca.explained_variance_ratio_
    pca_cum_sum = np.cumsum(pca_expl_var)
    if show_plots:
        plt.plot(pca_cum_sum)
        n_components = [i for i, mem in enumerate(pca_cum_sum.tolist())
                        if mem < 1 and mem > threshold][0]
        plt.plot([1 for i in pca_cum_sum])
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()
    return n_components

def get_NMF_explained_variance_ratio(x, ncomp):
    model = NMF(n_components=ncomp, init='random', random_state=0, max_iter=200)
    W = model.fit_transform(x)
    H = model.components_
    err = model.reconstruction_err_
    explained_var = 1 - np.square(err)/np.square(np.linalg.norm(x))
    return explained_var

def remove_nan_observations(x):
    """takes a np.array assuming that first dimension is the batch size or number of observations and remove
    any observation that has ALL nan features (cells with few nans and few non nans are kept).

        Args:
        x: a numpy array

        Returns:
        x : a numpy array with nan removed
        mask: a boolian array, True is showing non-nan values
        """
    shape = x.shape
    x_reshaped = x.reshape(shape[0], -1)
    # Drop all rows containing all nan:
    mask = np.isnan(x_reshaped).all(axis=1)
    x_reshaped = x_reshaped[~mask]
    # Reshape back:
    x = x_reshaped.reshape(x_reshaped.shape[0], *shape[1:])
    return x, ~mask


def convert_to_original_shape(x, mask):
    """takes a np.array and a mask array that has been used on the arr and return a original version of the arr

    Args:\
    x: a numpy array
    mask: a numpy array that was used to mask the original umased x

    Returns:
    x_unmasked
    """
    shape_as_list = list(x.shape)
    shape_as_list[0] = mask.shape[0]
    shape = tuple(shape_as_list)
    x_unmasked = np.zeros(shape)
    x_unmasked[mask] = x
    x_unmasked[~mask] = np.nan
    return x_unmasked


def report_mse_validation_losses_T_EM(outdict, train=True):
    '''
    From the model output computes the mse loss on all data
    Args:
        outmat: a dict with the cplAE output and input
    '''

    def allnans(x):
        return np.all(
            np.isnan(x.reshape(np.shape(x)[0], -1)), axis=(1))

    if train:
        ids = outdict["train"]
    else:
        ids = outdict["val"]

    # Masks
    m_T = ~allnans(outdict['XT'])[ids]
    m_E = ~allnans(outdict['XE'])[ids]
    m_M = ~allnans(outdict['XM'])[ids]
    m_TE_M = np.logical_and(m_T, np.logical_or(m_E, m_M))
    m_TE = np.logical_and(m_T, m_E)

    # input
    XT = outdict['XT'][ids]
    XE = outdict['XE'][ids]
    XM = outdict['XM'][ids]

    # within modality predictions
    XrT = outdict["XrT"][ids]
    XrE = outdict["XrE"][ids]
    XrM = outdict["XrM"][ids]

    # Embeddings
    zT = outdict["zT"][ids]
    zEM = outdict["zEM"][ids]

    mse_losses = {}
    mse_losses["XrT-XT"] = np.mean(np.square(XrT - XT))
    mse_losses["XrE-XE"] = np.nanmean(np.square(XrE - XE))
    mse_losses["XrM-XM"] = np.nanmean(np.square(XrM - XM))
    mse_losses["zEM-zT_for_TEM"] = np.mean(np.square(zEM[m_TE_M, :] - zT[m_TE_M, :]))
    mse_losses["zEM-zT_for_TE"] = np.mean(np.square(zEM[m_TE, :] - zT[m_TE, :]))

    return mse_losses


def get_umap_data(input):
    reducer = umap.UMAP()
    scaled_input = StandardScaler().fit_transform(input)
    output = reducer.fit_transform(scaled_input)
    return output


def load_exp_output(exp_name, pkl_file, results_folder="/home/fahimehb/Local/new_codes/cplAE_MET/data/results/"):
    '''Takes the exp_name and pkl file name and load the pkl file from the result folder'''
    output_folder = f"{results_folder}{exp_name}/"
    path = os.path.join(output_folder, pkl_file)
    return loadpkl(path)


def get_Leiden_comms(input):
    '''Takes model output and print the number of communities in each
    latent representation
    Args:
    input: This is the output .mat file from the coupled autoencoder run
    '''
    
    is_t_1d = input['is_t_1d']
    is_m_1d = input['is_m_1d']
    is_e_1d = input['is_e_1d']
    is_te_1d = np.logical_and(is_t_1d, is_e_1d)
    is_tm_1d = np.logical_and(is_t_1d, is_m_1d)
    is_me_1d = np.logical_and(is_m_1d, is_e_1d)
    is_met_1d = np.logical_and(is_t_1d, is_me_1d)

    df = pd.DataFrame(columns=["t_comm", "et_comm", "met_comm", "mt_comm"])
    df.loc[0, "t_comm"] = len(Leiden_community_detection(input['zt'][is_t_1d]).communities)
    df.loc[0, "et_comm"] = len(Leiden_community_detection(input['ze'][is_te_1d]).communities)
    df.loc[0, "met_comm"] = len(Leiden_community_detection(input['zme_paired'][is_met_1d]).communities)
    df.loc[0, "mt_comm"] = len(Leiden_community_detection(input['zm'][is_tm_1d]).communities)
    return df


def get_communities_sorted_node_labels(comm_obj):
    '''Get the community object and return the node labels for the 
    sorted nodes. This is required if we are going to pass a list of
    node lables to any algorithm
    Args:
    comm_obj: community detection object'''

    nodes = [i for i in comm_obj.to_node_community_map().keys()]
    node_labels = [i[0] for i in comm_obj.to_node_community_map().values()]
    myDict = {nodes[i]: node_labels[i] for i in range(len(nodes))}
    myKeys = list(myDict.keys())
    myKeys.sort()
    sorted_dict = {i: myDict[i] for i in myKeys}
    sorted_node_labels = [sorted_dict[k] for k in myKeys]

    return sorted_node_labels


def get_LDA_classification(output, level, reporting_on, train_inds, test_inds):
    '''Take the model output and the cluster label or the merged cluster labels and run LDA classification'''

    is_t_1d = output['is_t_1d']
    is_e_1d = output['is_e_1d']
    is_m_1d = output['is_m_1d']
    is_te_1d = np.logical_and(is_t_1d, is_e_1d)
    is_tm_1d = np.logical_and(is_m_1d, is_t_1d)
    is_met_1d = np.logical_and(is_m_1d, is_te_1d)
    
    if reporting_on=="kfold":
        X_train = output['zt'][is_t_1d]
        y_train = np.array([i.rstrip() for i in output[level][is_t_1d]])
        y = np.array([i.rstrip() for i in output[level]])
    
        kf = KFold(n_splits=10, random_state=None, shuffle=False)
        t_cpl_score = []
        te_cpl_score = []
        tm_cpl_score = []
        met_cpl_score = []
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            _, _, clf = run_LDA(X_train, y_train, train_test_ids= {'train': train_index, 'val': test_index})
            # This is the test acc on the t autoencoder
            t_cpl_score.append(clf.score(X_train[test_index], y_train[test_index]) * 100)
            # These are test acc on other autoencoders
            te_cpl_score.append(clf.score(output['ze'][is_te_1d], y[is_te_1d]) * 100)
            tm_cpl_score.append(clf.score(output['zm'][is_tm_1d], y[is_tm_1d]) * 100)
            met_cpl_score.append(clf.score(output['zme_paired'][is_met_1d], y[is_met_1d]) * 100)

        df = pd.DataFrame(columns=["t_clusters", "t_class_acc", "et_class_acc", "met_class_acc", "mt_class_acc"])
        df.loc[0, "t_clusters"] = len(np.unique(y_train))
        df.loc[0, "t_class_acc"] = "{:.2f}".format(np.mean(t_cpl_score))
        df.loc[0, "et_class_acc"] = "{:.2f}".format(np.mean(te_cpl_score))
        df.loc[0, "met_class_acc"] = "{:.2f}".format(np.mean(met_cpl_score))
        df.loc[0, "mt_class_acc"] = "{:.2f}".format(np.mean(tm_cpl_score))
    
    if reporting_on=="test_cells":
        assert (train_inds is not None)
        assert (test_inds is not None)
        is_train = np.array([True  if i in train_inds else False for i in range(len(is_t_1d))])
        is_test = np.array([True  if i in test_inds else False for i in range(len(is_t_1d))])
        
        re_ind_train = np.where(np.logical_and(is_train, is_t_1d))
        re_ind_test = np.where(np.logical_and(is_test, is_t_1d))
 

        X_train = output['zt']
        y_train = np.array([i.rstrip() for i in output[level]])
        y = np.array([i.rstrip() for i in output[level]])

        _, _, clf = run_LDA(X_train, y_train, train_test_ids= {'train': re_ind_train, 'val': re_ind_test})

        df = pd.DataFrame(columns=["t_clusters", "t_class_acc", "et_class_acc", "met_class_acc", "mt_class_acc"])
        df.loc[0, "t_clusters"] = len(np.unique(y_train))
        df.loc[0, "t_class_acc"] = "{:.2f}".format(clf.score(X_train[re_ind_test], y_train[re_ind_test]) * 100)
        df.loc[0, "et_class_acc"] = "{:.2f}".format(clf.score(output['ze'][test_inds], y[test_inds]) * 100)
        df.loc[0, "met_class_acc"] = "{:.2f}".format(clf.score(output['zme_paired'][test_inds], y[test_inds]) * 100)
        df.loc[0, "mt_class_acc"] = "{:.2f}".format(clf.score(output['zm'][test_inds], y[test_inds]) * 100)
        

    return df


def summary_classification_results(output, reporting_on="kfold", train_inds=None, test_inds=None):
    df = pd.concat([get_LDA_classification(output, level = "cluster_label", reporting_on=reporting_on, train_inds=train_inds, test_inds=test_inds),
                    get_LDA_classification(output, level = "merged_cluster_label_at40", reporting_on=reporting_on, train_inds=train_inds, test_inds=test_inds),
                    get_LDA_classification(output, level = "merged_cluster_label_at50", reporting_on=reporting_on, train_inds=train_inds, test_inds=test_inds)]
                    )
    return df


def summary_leiden_comm_silhouette_score(output):
    is_t_1d = output['is_t_1d']
    is_e_1d = output['is_e_1d']
    is_m_1d = output['is_m_1d']
    is_te_1d = np.logical_and(is_t_1d, is_e_1d)
    is_tm_1d = np.logical_and(is_m_1d, is_t_1d)
    is_met_1d = np.logical_and(is_m_1d, is_te_1d)
    
    df = pd.DataFrame(columns=["sil_score_t", "sil_score_et", "sil_score_met", "sil_score_mt"])

    for mode, mask, key in zip(["zt", "ze", "zme_paired", "zm"], 
                          [is_t_1d, is_te_1d, is_met_1d, is_tm_1d], 
                          ["sil_score_t", "sil_score_et", "sil_score_met", "sil_score_mt"]):
        comm = Leiden_community_detection(output[mode][mask])
        sorted_node_labels = get_communities_sorted_node_labels(comm)
        df.loc[0, key] = silhouette_score(output[mode][mask], sorted_node_labels) 
    
    return df


def summarize_data_output_pkl(exp_name, pkl_output_file):
    '''
    Takes the spec id locked dataset and returns the summary of the paltforms and modalities available
    Args:
    exp_name: name of the experiment folder in the results folder
    pkl_output_file: Name of the pkl file in the exp folder that should be loaded
    '''

    output = loadpkl("/home/fahimehb/Local/new_codes/cplAE_MET/data/results/" + exp_name + "/" + pkl_output_file)

    is_t_1d = output['is_t_1d']
    is_e_1d = output['is_e_1d']
    is_m_1d = output['is_m_1d']
    is_te_1d = np.logical_and(is_t_1d, is_e_1d)
    is_tm_1d = np.logical_and(is_t_1d, is_m_1d)
    is_me_1d = np.logical_and(is_m_1d, is_e_1d)
    is_met_1d = np.logical_and(is_e_1d, is_tm_1d)
    is_t_only = np.logical_and(is_t_1d, np.logical_and(~is_e_1d, ~is_m_1d))
    is_e_only = np.logical_and(is_e_1d, np.logical_and(~is_t_1d, ~is_m_1d))
    is_m_only = np.logical_and(is_m_1d, np.logical_and(~is_e_1d, ~is_t_1d))
    is_te_only = np.logical_and(is_te_1d, ~is_m_1d)
    is_me_only = np.logical_and(is_me_1d, ~is_t_1d)
    is_tm_only = np.logical_and(is_tm_1d, ~is_e_1d)


    if "platform" in output.keys():
        # output["platform"] = [i.rstrip() for i in output["platform"]]
        plats = np.unique(output["platform"])

        summary1 = pd.DataFrame(columns=["platform", "T", "E", "M", "E&T", "M&T", "M&E", "M&E&T", "total"])
        for i, p in enumerate(plats):
            platform_mask = np.array([True if i==p else False for i in output["platform"]])
            summary1.loc[i, "platform"] = p
            summary1.loc[i, "T"] = np.sum(np.logical_and(platform_mask, is_t_only))
            summary1.loc[i, "E"] = np.sum(np.logical_and(platform_mask, is_e_only))
            summary1.loc[i, "M"] = np.sum(np.logical_and(platform_mask, is_m_only))
            summary1.loc[i, "E&T"] = np.sum(np.logical_and(platform_mask, is_te_only))
            summary1.loc[i, "M&T"] = np.sum(np.logical_and(platform_mask, is_tm_only))
            summary1.loc[i, "M&E"] = np.sum(np.logical_and(platform_mask, is_me_only))
            summary1.loc[i, "M&E&T"] = np.sum(np.logical_and(platform_mask, is_met_1d))
            summary1.loc[i, "total"] = int(platform_mask.sum())
        summary1.loc[4,"platform"] = "all"
        summary1.loc[4, ["T", "E", "M", "E&T", "M&T", "M&E", "M&E&T", "total"]]= summary1.sum(axis=0).to_list()

    if "class" in output.keys():
        output["class"] = [i.rstrip() for i in output["class"]]
        summary2 = pd.DataFrame(columns=["class", "T", "E", "M", "E&T", "M&T", "M&E", "M&E&T", "total"])
        for i, c in enumerate(['exc', 'inh']):
            class_mask = np.array([True if i == c else False for i in output["class"]])
            summary2.loc[i, "class"] = c
            summary2.loc[i, "T"] = np.sum(np.logical_and(class_mask, is_t_only))
            summary2.loc[i, "E"] = np.sum(np.logical_and(class_mask, is_e_only))
            summary2.loc[i, "M"] = np.sum(np.logical_and(class_mask, is_m_only))
            summary2.loc[i, "E&T"] = np.sum(np.logical_and(class_mask, is_te_only))
            summary2.loc[i, "M&T"] = np.sum(np.logical_and(class_mask, is_tm_only))
            summary2.loc[i, "M&E"] = np.sum(np.logical_and(class_mask, is_me_only))
            summary2.loc[i, "M&E&T"] = np.sum(np.logical_and(class_mask, is_met_1d))
            summary2.loc[i, "total"] = int(class_mask.sum())
        summary2.loc[2,"class"] = "all"
        summary2.loc[2, ["T", "E", "M", "E&T", "M&T", "M&E", "M&E&T", "total"]]= summary2.sum(axis=0).to_list()[1:]
        return summary1, summary2
    else:
        return summary1
    

def summarize_data_input_mat(exp_name, pkl_output_file):
    '''
    Takes the spec id locked dataset and returns the summary of the paltforms and modalities available
    Args:
    exp_name: name of the experiment folder in the results folder
    pkl_output_file: Name of the pkl file in the exp folder that should be loaded
    ''' 
    output = loadpkl("/home/fahimehb/Local/new_codes/cplAE_MET/data/results/" + exp_name + "/" + pkl_output_file)

    is_t_1d = output['is_t_1d']
    is_e_1d = output['is_e_1d']
    is_m_1d = output['is_m_1d']
    is_te_1d = np.logical_and(is_t_1d, is_e_1d)
    is_tm_1d = np.logical_and(is_t_1d, is_m_1d)
    is_me_1d = np.logical_and(is_m_1d, is_e_1d)
    is_met_1d = np.logical_and(is_e_1d, is_tm_1d)
    is_t_only = np.logical_and(is_t_1d, np.logical_and(~is_e_1d, ~is_m_1d))
    is_e_only = np.logical_and(is_e_1d, np.logical_and(~is_t_1d, ~is_m_1d))
    is_m_only = np.logical_and(is_m_1d, np.logical_and(~is_e_1d, ~is_t_1d))
    is_te_only = np.logical_and(is_te_1d, ~is_m_1d)
    is_me_only = np.logical_and(is_me_1d, ~is_t_1d)
    is_tm_only = np.logical_and(is_tm_1d, ~is_e_1d)

    output["platform"] = [i.rstrip() for i in output["platform"]]

    summary1 = pd.DataFrame(columns=["platform", "T", "E", "M", "E&T", "M&T", "M&E", "M&E&T", "total"])
    for i, p in enumerate(["patchseq", "ME", "EM", "fMOST"]):
        platform_mask = np.array([True if i==p else False for i in output["platform"]])
        summary1.loc[i, "platform"] = p
        summary1.loc[i, "T"] = np.sum(np.logical_and(platform_mask, is_t_only))
        summary1.loc[i, "E"] = np.sum(np.logical_and(platform_mask, is_e_only))
        summary1.loc[i, "M"] = np.sum(np.logical_and(platform_mask, is_m_only))
        summary1.loc[i, "E&T"] = np.sum(np.logical_and(platform_mask, is_te_only))
        summary1.loc[i, "M&T"] = np.sum(np.logical_and(platform_mask, is_tm_only))
        summary1.loc[i, "M&E"] = np.sum(np.logical_and(platform_mask, is_me_only))
        summary1.loc[i, "M&E&T"] = np.sum(np.logical_and(platform_mask, is_met_1d))
        summary1.loc[i, "total"] = int(platform_mask.sum())
    summary1.loc[4,"platform"] = "all"
    summary1.loc[4, ["T", "E", "M", "E&T", "M&T", "M&E", "M&E&T", "total"]]= summary1.sum(axis=0).to_list()[1:]

    if "class" in output.keys():
        output["class"] = [i.rstrip() for i in output["class"]]
        summary2 = pd.DataFrame(columns=["class", "T", "E", "M", "E&T", "M&T", "M&E", "M&E&T", "total"])
        for i, c in enumerate(['exc', 'inh']):
            class_mask = np.array([True if i == c else False for i in output["class"]])
            summary2.loc[i, "class"] = c
            summary2.loc[i, "T"] = np.sum(np.logical_and(class_mask, is_t_only))
            summary2.loc[i, "E"] = np.sum(np.logical_and(class_mask, is_e_only))
            summary2.loc[i, "M"] = np.sum(np.logical_and(class_mask, is_m_only))
            summary2.loc[i, "E&T"] = np.sum(np.logical_and(class_mask, is_te_only))
            summary2.loc[i, "M&T"] = np.sum(np.logical_and(class_mask, is_tm_only))
            summary2.loc[i, "M&E"] = np.sum(np.logical_and(class_mask, is_me_only))
            summary2.loc[i, "M&E&T"] = np.sum(np.logical_and(class_mask, is_met_1d))
            summary2.loc[i, "total"] = int(class_mask.sum())
        summary2.loc[2,"class"] = "all"
        summary2.loc[2, ["T", "E", "M", "E&T", "M&T", "M&E", "M&E&T", "total"]]= summary2.sum(axis=0).to_list()[1:]
        return summary1, summary2
    else:
        return summary1


def Get_available_modalities_in_each_platform(input_mat_file):
    '''Takes the input mat file name and gets all the available modalities in each platform'''
    
    mat = sio.loadmat("/home/fahimehb/Remote-AI-root/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Patchseq-Exc/dat/" + input_mat_file)

    is_t_1d = np.any(~np.isnan(mat['T_dat']), axis=1)
    is_e_1d = np.any(~np.isnan(mat['E_dat']), axis=1)
    is_m_1d = np.any(~np.isnan(mat['M_dat']), axis=(1,2,3))
    is_te_1d = np.logical_and(is_t_1d, is_e_1d)
    is_me_1d = np.logical_and(is_m_1d, is_e_1d)
    is_tm_1d = np.logical_and(is_t_1d, is_m_1d)
    is_tem_1d = np.logical_and(is_te_1d, is_m_1d)
    mat['platform'] = np.array([i.rstrip() for i in mat['platform']])
    is_patchseq_1d = mat['platform']=="patchseq"
    is_ME_1d = mat['platform']=="ME"
    is_fMOST_1d = mat['platform']=="fMOST"
    is_EM_1d = mat['platform']=="EM"
    is_patchseq_tem_1d = np.logical_and(is_patchseq_1d, is_tem_1d)
    is_patchseq_teonly_1d = np.logical_and(~is_m_1d, np.logical_and(is_patchseq_1d, is_te_1d))
    is_patchseq_tmonly_1d = np.logical_and(~is_e_1d, np.logical_and(is_patchseq_1d, is_tm_1d))
    is_patchseq_meonly_1d = np.logical_and(~is_t_1d , np.logical_and(is_patchseq_1d, is_me_1d))
    is_patchseq_tonly_1d = np.logical_and(~is_m_1d , np.logical_and(is_patchseq_1d, ~is_e_1d))
    is_patchseq_eonly_1d = np.logical_and(~is_m_1d , np.logical_and(is_patchseq_1d, ~is_t_1d))
    is_patchseq_monly_1d = np.logical_and(~is_t_1d , np.logical_and(is_patchseq_1d, ~is_e_1d))
    is_ME_eonly_1d =  np.logical_and(is_ME_1d, ~is_m_1d)
    is_ME_monly_1d =  np.logical_and(is_ME_1d, ~is_e_1d)
    is_ME_me_1d = np.logical_and(is_ME_1d, is_me_1d) 

    print("cells with t data available", is_t_1d.sum()) # These come from patchseq data
    print("cells with e data available", is_e_1d.sum()) # these come from ME and patchseq data
    print("cells with m data available", is_m_1d.sum()) # these come from patchseq, fmost, ME and EM data
    print("cells with te data available", is_te_1d.sum()) # these come from patchseq
    print("cells with me data available", is_me_1d.sum()) # these come from patchseq and ME
    print("cells with tm data available", is_tm_1d.sum()) # these come from patchseq
    print("cells with tem data available", is_tem_1d.sum()) # these come from patchseq
    print("patchseq cells:", is_patchseq_1d.sum())
    print("patchseq cells with all t,e,m:", is_patchseq_tem_1d.sum())
    print("patchseq cells with only t,e:", is_patchseq_teonly_1d.sum()) 
    print("patchseq cells with only t,m:", is_patchseq_tmonly_1d.sum()) 
    print("patchseq cells with only m,e:", is_patchseq_meonly_1d.sum()) 
    print("patchseq cells with only t:", is_patchseq_tonly_1d.sum()) 
    print("patchseq cells with only e:", is_patchseq_eonly_1d.sum())
    print("patchseq cells with only m:", is_patchseq_monly_1d.sum())
    print("ME cells:", is_ME_1d.sum())
    print("ME cells with only m:",is_ME_monly_1d.sum())
    print("ME cells with only e:",is_ME_eonly_1d.sum())
    print("ME cells with both e, m:",is_ME_me_1d.sum())
    print("total number of cells:", is_patchseq_1d.sum() + is_EM_1d.sum() + is_ME_1d.sum() + is_fMOST_1d.sum())
    print("check number of t cells:", is_t_1d.sum() ==
        (is_patchseq_tem_1d.sum() + is_patchseq_teonly_1d.sum() + is_patchseq_tmonly_1d.sum() + is_patchseq_tonly_1d.sum()))
    print("check number of e cells:", is_e_1d.sum() == 
        (is_patchseq_tem_1d.sum() + is_patchseq_teonly_1d.sum()+ is_patchseq_meonly_1d.sum() + is_patchseq_eonly_1d.sum() + is_ME_eonly_1d.sum() + is_ME_me_1d.sum()))
    print("check number of m cells:", is_m_1d.sum() ==
        (is_patchseq_tem_1d.sum() + is_patchseq_tmonly_1d.sum() + is_patchseq_meonly_1d.sum() + is_patchseq_monly_1d.sum() + is_ME_monly_1d.sum() + is_ME_me_1d.sum() + 
        is_fMOST_1d.sum() + is_EM_1d.sum()))
    print("Note that all EM and fMOST cells are m only cells")
    print("Number of EM cells:", is_EM_1d.sum())
    print("Number of fMOST cells:", is_fMOST_1d.sum())


    return is_patchseq_tem_1d.sum(), is_patchseq_teonly_1d.sum(), is_patchseq_tmonly_1d.sum(), \
           is_patchseq_meonly_1d.sum(), is_patchseq_tonly_1d.sum(),  is_patchseq_eonly_1d.sum(), \
           is_patchseq_monly_1d.sum(), is_ME_monly_1d.sum(), is_ME_eonly_1d.sum(), is_ME_me_1d.sum(), \
           is_EM_1d.sum(), is_fMOST_1d.sum()


def calculate_and_plot_umap(exp_name, pkl_file, ttype_resolution="merged_cluster_label_at60", autoencoder_arm_embedding_key="zm"):
    '''Takes the experiment name and the optimal output pkl file, as well as the resolution
    level for ttypes. Then removes the cells that blong to small clusters and for the remaining
    cells computes the umap for the requested autoencoder arm
    Arg:
       exp_name: exp folder name, all the experiments are located in the results folder in the cplae package
       pkl_file: name of the pkl file that was obtained during the optimization of cplae run
       ttype_resolution: resolution of the ttypes taxonomy that should be used, for example merged_cluster_label_at60
       autoencoder_arm_embedding_key: which embedding to be used, for example: zt, ze, zme_paired or zm
    '''
    output = load_exp_output(exp_name=exp_name, pkl_file=pkl_file)
    is_m_1d = output['is_m_1d']
    is_e_1d = output['is_e_1d']
    is_t_1d = output['is_t_1d']
    is_tm_1d = np.logical_and(is_t_1d, is_m_1d)
    is_te_1d = np.logical_and(is_t_1d, is_e_1d)
    is_tem_1d = np.logical_and(is_e_1d, is_tm_1d)
    
    if autoencoder_arm_embedding_key == "zm":
        mask = is_tm_1d
    if autoencoder_arm_embedding_key == "zme_paired":
        mask = is_tem_1d
    if autoencoder_arm_embedding_key == "ze":
        mask = is_te_1d
    if autoencoder_arm_embedding_key == "zt":
        mask = is_t_1d

    labels = np.array([i.rstrip() for i in output[ttype_resolution][mask]])
    
    # so we are going to drop all the cells from the clusters that have less than 7 membs
    drop_lables = []
    for k,v in Counter(labels).items():
        if v < 7 :
            drop_lables.append(k)

    drop_small_clusters_mask = [True if i not in drop_lables else False for i in labels]

    X = output[autoencoder_arm_embedding_key][mask][drop_small_clusters_mask]
    y = output[ttype_resolution][mask][drop_small_clusters_mask]
    colors = output['cluster_color'][mask][drop_small_clusters_mask]
    Xid = np.arange(len(output['specimen_id']))[mask][drop_small_clusters_mask]

    #Since we merged t-type, we need to consolidate the t-colors too
    updated_colors = colors

    for c in Counter(y):
        cc = []
        for i,j in zip(y, colors):
            if i == c:
                cc.append(j)
        new_color = max(Counter(cc))
        updated_colors = [new_color if k==c else l for (k,l) in zip(y, updated_colors)]

    print(X.shape, y.shape)

    umap_x = umap.UMAP().fit_transform(X)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.scatter(umap_x[:,0], umap_x[:,1], color=updated_colors,s=3)
    plt.show()
    return X, Xid, y, umap_x, updated_colors, fig

