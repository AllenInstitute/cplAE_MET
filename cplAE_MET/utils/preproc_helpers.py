import json 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from cplAE_TE.utils.load_helpers import get_paths


def check_for_nan(nparray):
    """Takes a nparray and check if there is any nan values in it.

    Args:
    nparray: a numpy array

    Returns:
    True or False
    """
    return np.sum(np.all(np.isnan(nparray), axis=1)) > 0


def return_nan_index_col(nparray):
    """Takes a nparray and return the index and col of all nan values

    Args:
    nparray: a numpy array

    Returns:
    index_list: list of all index values of nans
    col_list: list of all col values of nans
    """
    index_list = np.where(np.isnan(x))[0]
    col_list = np.where(np.isnan(x))[1]
    return index_list, col_list


def drop_nan_rows_or_cols(nparray, axis):
    """Takes a nparray and drop rows or cols with nan values

    Args:
    nparray: a numpy array

    Returns:
    final_nparray: a numpy array
    dropped: a list of dropped indices or cols
    """
    mask = np.all(np.isnan(nparray), axis=axis)
    final_nparray = nparray[~mask]
    dropped = np.where(mask)[0]
    return final_nparray, dropped


def get_PCA_explained_variance_ratio_at_thr(nparray, threshold):
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
    plt.plot(pca_cum_sum)
    n_components = [i for i, mem in enumerate(pca_cum_sum.tolist())
                    if mem < 1 and mem > threshold][0]
    plt.plot([1 for i in pca_cum_sum])
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    return n_components
