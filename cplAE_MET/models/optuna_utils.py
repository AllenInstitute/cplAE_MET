import torch
import numpy as np
# For community detection
import networkx as nx
from cdlib import algorithms
from sklearn.neighbors import kneighbors_graph

from cplAE_MET.models.torch_utils import tonumpy
from cplAE_MET.models.classification_functions import run_LDA


def run_classification(model, dataloader, train_ind, val_ind, T_labels):
        '''
        Take the model and the dataloader, to run the model in the evaluation mode, obtain the emd 
        and run the classisification task on the t cells'''

        z_dict = {"zt": [], "ze": [], "zm": [], "zme_paired": []}
        is_t_1d = []

        model.eval()
        for i, batch in enumerate(iter(dataloader)):
            with torch.no_grad():

                is_t_1d.append(batch['is_t_1d'])

                _, z_di, _ = model(batch) 

                z_dict["zt"].append(z_di["zt"])
                z_dict["ze"].append(z_di["ze"])
                z_dict["zm"].append(z_di["zm"])
                z_dict["zme_paired"].append(z_di["zme_paired"])


        is_t_1d = torch.cat(is_t_1d)
        z_dict["zt"] = torch.cat(z_dict["zt"])
        z_dict["ze"] = torch.cat(z_dict["ze"])
        z_dict["zm"] = torch.cat(z_dict["zm"])
        z_dict["zme_paired"] = torch.cat(z_dict["zme_paired"])

        is_t_1d = tonumpy(is_t_1d)
        is_train_1d = np.array([False for i in (range(len(is_t_1d)))])
        is_train_1d[train_ind] = True
        is_test_1d = np.array([False for i in (range(len(is_t_1d)))])
        is_test_1d[val_ind] = True 
        is_t_train_1d = np.logical_and(is_t_1d, is_train_1d)
        is_t_test_1d = np.logical_and(is_t_1d, is_test_1d)
        t_train_ind = np.where(is_t_train_1d)
        t_test_ind = np.where(is_t_test_1d)

        zt = tonumpy(z_dict['zt'])
        ze = tonumpy(z_dict['ze'])
        zm = tonumpy(z_dict['zm'])
        zme_paired = tonumpy(z_dict['zme_paired'])

        _, _, clf = run_LDA(zt, 
                            T_labels,
                            train_test_ids={'train': t_train_ind, 
                                            'val': t_test_ind})
        
        te_cpl_score = clf.score(ze[val_ind], T_labels[val_ind]) * 100
        tm_cpl_score = clf.score(zm[val_ind], T_labels[val_ind]) * 100
        met_cpl_score = clf.score(zme_paired[val_ind], T_labels[val_ind]) * 100
        print("te, tm and met classification acc:", te_cpl_score, tm_cpl_score, met_cpl_score)
        return np.min([te_cpl_score, tm_cpl_score, met_cpl_score])


def Leiden_community_detection(data):
    '''Take the data points, creat a graph with 12 nn and apply leiden community detection
    Args:
        data: the embedding coordinates
    '''
    # Create adj matrix with 12 nn
    A = kneighbors_graph(data, 12, mode='distance', include_self=True)
    # Create a network_x graph
    G = nx.convert_matrix.from_numpy_array(A)
    # Run Leiden community detection algorithm
    comm = algorithms.leiden(G)
    return comm


def run_Leiden_community_detection(model, dataloader):
    '''
    Takes the model and dataloader to run the model in the evaluation mode and compute
    the number of leiden communities on the embedding
    '''
    model.eval()
    for all_data in iter(dataloader):
        _, z_dict, _ = model(all_data) 

    is_t_1d = tonumpy(all_data['is_t_1d'])
    is_e_1d = tonumpy(all_data['is_e_1d'])
    is_m_1d = tonumpy(all_data['is_m_1d'])
    is_me_1d = np.logical_and(is_m_1d, is_e_1d)
    is_met_1d = np.logical_and(is_t_1d, is_me_1d)

    zt = tonumpy(z_dict['zt'])
    zme_paired = tonumpy(z_dict['zme_paired'])
    
    n_t_types = []
    n_me_types = []
    # Instead of running it only once, we run it 10 times and then take the max
    for i in range(10):
        ncomm = len(Leiden_community_detection(zt[is_t_1d]).communities)
        n_t_types.append(ncomm)
        ncomm = len(Leiden_community_detection(zme_paired[is_met_1d]).communities)
        n_me_types.append(ncomm)

    n_t_types = np.median(n_t_types)
    n_me_types = np.median(n_me_types)
    model_score = np.min([n_t_types , n_me_types])
    return model_score