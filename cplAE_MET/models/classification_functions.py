import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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


def run_LogisticRegression(X, y, test_size, min_label_size=7):
    '''

    Args:
        X: input array with the size of (n_cells, n_features)
        y: labels for each X entry with the size of (n_cells, )
        test_size: float value for the split
        min_label_size: all clusters with less than min_label_size number of members will be removed

    Returns:
        accuracy of the classification task

    '''
    small_types_mask = get_small_types_mask(y, min_label_size)
    X = X[small_types_mask]
    _, y = np.unique(y[small_types_mask], return_inverse=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=0)

    clf = LogisticRegression(random_state=0, max_iter=10000).fit(X_train, y_train)
    classification_acc = clf.score(X_test, y_test) * 100
    n_class = len(Counter(y_test))
    return classification_acc, n_class


def run_QDA(X, y, test_size, min_label_size=7, train_test_ids=None):
    '''

    Args:
        X: input array with the size of (n_cells, n_features)
        y: labels for each X entry with the size of (n_cells, )
        test_size: float value for the split
        min_label_size: all clusters with less than min_label_size number of members will be removed
        train_test_ids: the indices of the train and test cells to be used in the classifier
    Returns:
        accuracy of the classification task, number of classes and the classifier object

    '''
    # We need to remove the cells that are within T types with small number of cells
    small_types_mask = get_small_types_mask(y, min_label_size)

    if train_test_ids:
        # first we remove the small type cells from the train and test/val cells
        train_mask = [i for i in train_test_ids['train'] if i in np.where(small_types_mask)[0]]
        test_mask = [i for i in train_test_ids['val'] if i in np.where(small_types_mask)[0]]

        # apply the mask
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        # we need to factorize the y as they are not numbers but are cluster labels
        y_train_test = np.append(y_train, y_test)
        _, y_train_test = np.unique(y_train_test, return_inverse=True)
        y_test = y_train_test[-len(y_test):]
        y_train = y_train_test[: len(y_train)]
    else:
        X = X[small_types_mask]
        y = y[small_types_mask]
        _, y = np.unique(y, return_inverse=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=0)

    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    classification_acc = clf.score(X_test, y_test) * 100
    n_class = len(Counter(y_test))
    return classification_acc, n_class, clf
