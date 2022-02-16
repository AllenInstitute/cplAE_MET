import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import cplAE_MET.utils.utils as ut
from cplAE_MET.utils.utils import loadpkl
from os import system, name
from time import sleep

def get_exit_summary_keys(listdir):
    '''
    Get list of exit_summary files in a folder and return the hyperparameters used in the run
    Args:
        listdir:

    Returns:

    '''
    vars = []
    for f in listdir:
        if f.endswith(".pkl"):
            model_id = re.search('summary_(.*)_aM_',f).group(1)
            aM = re.search('aM_(.*)_asd_',f).group(1)
            asd = re.search('asd_(.*)_noise_',f).group(1)
            noise = re.search('noise_(.*)_dilate_',f).group(1)
            scale = re.search('scale_(.*)_ld_',f).group(1)
            ld = re.search('ld_(.*)_ne_', f).group(1)
            fold = re.search('fold_(.*).pkl',f).group(1)
            vars.append((model_id, aM, asd, noise, scale, ld, fold))
    return vars


def get_exit_summary_df(dir, file_keys):
    '''
    takes the path of the files and find the hyperparam values of each run and return them in a datafram format
    Args:
        dir:
        file_keys:

    Returns:

    '''
    output = {}
    df = pd.DataFrame(columns=["model_id", "aM", "asd", "noise", "scale", "ld", "fold", "classification_acc"])
    for var in file_keys:
        fileid = "exit_summary_" + var[0] + "_aM_" + var[1] + "_asd_" + var[2] + "_noise_" + var[
            3] + "_dilate_0_scale_" + var[4] + "_ld_" + str(var[5]) + "_ne_50000_ri_0_fold_" + str(var[6]) + ".pkl"

        path = dir + fileid

        output[var] = loadpkl(path)

        df = df.append({'model_id': var[0],
                        "aM": var[1].replace("-", "."),
                        "asd": var[2].replace("-", "."),
                        "noise": var[3].replace("-", "."),
                        "scale": var[4].replace("-", "."),
                        "ld": var[5],
                        "fold": var[6],
                        "classification_acc": output[var]['classification_acc']}, ignore_index=True)

    return output, df

def summarize_folder(path):
    '''
    Take the path and return the output of all the runs as a dict and the hyperparams as a dataframe
    Args:
        path:

    Returns:

    '''
    files = os.listdir(path)
    keys = get_exit_summary_keys(files)
    output, df = get_exit_summary_df(path, keys)
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

def summarize_model_folder(directory):

    summary = pd.DataFrame(columns=["Alpha_T", "Alpha_E", "Alpha_M", "Lambda_T_EM", "M_noise", "E_noise", "n_fold",
                                    "mse(XT-XrT)_train", "mse(XT-XrT)_val",
                                    "mse(XE-XrE)_train", "mse(XE-XrE)_val",
                                    "mse(XM-XrM)_train", "mse(XM-XrM)_val",
                                    "mse(zEM-zT_for_TEM)_train", "mse(zEM-zT_for_TEM)_val",
                                    "mse(zEM-zT_for_TE)_train", "mse(zEM-zT_for_TE)_val"])

    i = 0
    output_mat = {}
    for f in os.listdir(directory):
        # clear_output(wait=True)
        # print("file number" ,i, "is being loaded")
        if f.endswith(".pkl"):
            fileid = re.search('(.*)_exit-summary', f).group(1)
            alpha_T = float(re.search('aT_(.*)_aE', f).group(1).replace("-", "."))
            alpha_E = float(re.search('aE_(.*)_aM', f).group(1).replace("-", "."))
            alpha_M = float(re.search('aM_(.*)_asd', f).group(1).replace("-", "."))
            lambda_T_EM = float(re.search('csT_EM_(.*)_ad', f).group(1).replace("-", "."))
            M_noise = float(re.search('_Mnoi_(.*)_Enoi', f).group(1).replace("-", "."))
            # E_noise = float(re.search('_Enoi_(.*)__dil_M_', f).group(1).replace("-", "."))
            n_fold = float(re.search('_fold_(.*)_exit', f).group(1).replace("-", "."))

            # print(fileid)

            pth = os.path.join(directory, f)
            # checking if it is a file
            if os.path.isfile(pth):
                output_mat[fileid] = ut.loadpkl(pth)
                MSE_train = report_mse_validation_losses_T_EM(output_mat[fileid], train=True)
                MSE_val = report_mse_validation_losses_T_EM(output_mat[fileid], train=False)

                summary.loc[i] = [alpha_T, alpha_E, alpha_M, lambda_T_EM, M_noise, E_noise, n_fold,
                                  MSE_train["XrT-XT"], MSE_val["XrT-XT"],
                                  MSE_train["XrE-XE"], MSE_val["XrE-XE"],
                                  MSE_train["XrM-XM"], MSE_val["XrM-XM"],
                                  MSE_train["zEM-zT_for_TEM"], MSE_val["zEM-zT_for_TEM"],
                                  MSE_train["zEM-zT_for_TE"], MSE_val["zEM-zT_for_TE"]]

                i+=1
    return summary


