import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA



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
    return [r2_score(X[:, i], Xr[:, i]) for i in which_cols]


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