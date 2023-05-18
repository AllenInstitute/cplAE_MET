# %%
import os
import argparse
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from collections import Counter

from imblearn.ensemble import BalancedRandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--em_cell_index',           default=0,  type=int,   help='Index of EM cell')
parser.add_argument('--n_classification_iter',   default=1,  type=int,   help='numbner of classification iteration')
parser.add_argument('--output_file_path',        default='/home/fahimehb/Local/new_codes/cplAE_MET/data/results/classification_results/',         type=str,   help='Experiment set')


def main(em_cell_index=0,
         n_classification_iter=1,
         output_file_path='TEMP'):


    # %%
    path = "/home/fahimehb/Remote-AI-root/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Patchseq-Exc/dat/"
    mass_normalized_matfile = "MET_200nmfs_2channels_10May23.mat"
    dat = sio.loadmat(path + mass_normalized_matfile)
    dat.keys()

    #%%
    is_m_1d = np.any(~np.isnan(dat['M_dat']), axis=1)
    summary = pd.DataFrame(columns=['platform', 'cluster_label', 'is_m_1d'])
    summary['platform'] = [i.rstrip() for i in dat['platform']]
    summary['cluster_label'] = [i.rstrip() for i in dat['cluster_label']]
    summary['class'] = dat['class']
    summary['is_m_1d'] = is_m_1d
    summary['specimen_id'] = [i.rstrip() for i in dat['specimen_id']]

    #%%
    # Keep only data that has m 
    m_data = dat['M_dat'][is_m_1d]
    summary = summary.loc[summary['is_m_1d']].reset_index().drop(columns=['index'])
    summary

    # %%
    print(m_data.shape)
    print(summary.shape)
    #Mask only the M_cells 
    #set the lables for EM M cells to 1 and for the other M cells to 0
    summary['label'] = 0
    summary.loc[(summary['platform']=="EM"), 'label'] = 1
    Counter(summary['label'])

    #%%
    summary['EM_train_cells'] = False
    summary['patchseq_train_cells'] = False
    summary.loc[(summary['platform']=="EM"), "EM_train_cells"] = True
    summary.loc[(summary['platform']!="EM"), "patchseq_train_cells"] = True
    summary
    print(Counter(summary['EM_train_cells']))
    print(Counter(summary['patchseq_train_cells']))

    # %%
    EM_train_cells_ind = np.where(np.array(summary['EM_train_cells'].to_list()))[0]
    patchseq_train_cells_ind = np.where(np.array(summary['patchseq_train_cells'].to_list()))[0]
    train_ind = np.concatenate((EM_train_cells_ind, patchseq_train_cells_ind))

    #%%
    X = m_data
    y = np.array([0 for i in range(X.shape[0])])
    y[summary['EM_train_cells'].to_list()]=1

    # %%
    select_cell = EM_train_cells_ind[em_cell_index]
    test_ind = [select_cell]
    new_train_ind = [j for j in train_ind if j!=select_cell]

    X_train = X[new_train_ind]
    X_test = X[test_ind]

    y_train = y[new_train_ind]
    y_test = y[test_ind]
    acc = []
    df = pd.DataFrame(columns=["em_cell", "mean_calssification_acc"])
    print("Running classification for this cell:", select_cell)
    for run in range(n_classification_iter):
        model = BalancedRandomForestClassifier(n_estimators=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc.append(y_pred==y_test)
        df.loc[0, "mean_calssification_acc"] = np.mean(acc) 
        df.loc[0, "em_cell"] = summary.loc[select_cell,'specimen_id']  
    if os.path.exists("output.txt"):
        os.remove("output.txt")
    df.to_csv(output_file_path + str(em_cell_index)+ ".csv", index=None)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
