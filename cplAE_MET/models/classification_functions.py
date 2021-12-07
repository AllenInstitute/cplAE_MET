import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def get_small_types_mask(types, min_size):
    '''
    Remove all the cells within types with small size
    Args:
        types: cell type labels array with the size of (n_cells,)
        min_size: the minimum size type to discard
    Returns:
        1D binary mask

    '''
    type_counts = Counter(types)
    type_rm = [k for k, v in type_counts.items() if v < min_size]
    return np.array([False if t in type_rm else True for t in types])


def run_LogisticRegression(X, y, stratify, test_size):
    '''

    Args:
        X: input array with the size of (n_cells, n_features)
        y: labels for each X entry with the size of (n_cells, )
        stratify: the column to be used for startifed split
        test_size: float value for the split

    Returns:
        accuracy of the classification task

    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=stratify, test_size=test_size, random_state=0)
    clf = LogisticRegression(random_state=0, max_iter=10000).fit(X_train, y_train)
    return clf.score(X_test, y_test) * 100