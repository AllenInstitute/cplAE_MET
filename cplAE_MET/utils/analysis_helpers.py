import os
import re
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import cplAE_MET.utils.utils as ut
from sklearn.preprocessing import StandardScaler



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


