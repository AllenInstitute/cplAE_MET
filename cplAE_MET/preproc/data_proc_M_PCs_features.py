import scipy.io as sio
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from functools import reduce
from cplAE_MET.utils.load_config import load_config
import cplAE_MET.utils.analysis_helpers as proc_utils

parser = argparse.ArgumentParser()
parser.add_argument('--config_file',    default='config_preproc.toml', type=str,   help='config file with data paths')
parser.add_argument('--pca_th',     default=0.90,  type=float,   help='threshold for pca component')

def set_paths(config_file=None):
    paths = load_config(config_file=config_file, verbose=False)

    paths['input'] = f'{str(paths["data_dir"])}'
    paths['arbor_density_file'] = f'{paths["input"]}/{str(paths["arbor_density_file"])}'
    paths['arbor_density_PC_file'] = f'{paths["input"]}/{str(paths["arbor_density_PC_file"])}'
    paths['arbor_density_PC_and_features_file'] = f'{paths["input"]}/{str(paths["arbor_density_PC_and_features_file"])}'
    paths['arbor_density_PC_vars_file'] = f'{paths["input"]}/{str(paths["arbor_density_PC_vars_file"])}'
    paths['ivscc_inhib_RawFeatureWide'] = f'{paths["input"]}/{str(paths["ivscc_inhib_RawFeatureWide"])}'
    paths['ivscc_exc_RawFeatureWide'] = f'{paths["input"]}/{str(paths["ivscc_exc_RawFeatureWide"])}'
    paths['fmost_exc_RawFeatureWide'] = f'{paths["input"]}/{str(paths["fmost_exc_RawFeatureWide"])}'


    paths['specimen_ids'] = f'{paths["input"]}/{str(paths["specimen_ids_file"])}'

    paths['E_timeseries'] = f'{paths["input"]}/{str(paths["E_timeseries_file"])}'
    paths['ipfx_features'] = f'{paths["input"]}/{str(paths["ipfx_features_file"])}'
    paths['e_output'] = f'{paths["input"]}/{str(paths["e_output_file"])}'

    return paths

def main(config_file='config_preproc.toml', pca_th=0.97):

    dir_pth = set_paths(config_file=config_file)
    ids = pd.read_csv(dir_pth['specimen_ids'])
    specimen_ids = ids['specimen_id'].tolist()
    print("...................................................")
    print("There are", len(specimen_ids), "sample_ids in the locked dataset")


    print("...................................................")
    print("Loading M arbor_densities")
    m_input = sio.loadmat(dir_pth['arbor_density_file'])

    print("...................................................")
    print("Loading m features")
    # read M_metadata 
    ivscc_inh = pd.read_csv(dir_pth['ivscc_inhib_RawFeatureWide'])
    ivscc_exc = pd.read_csv(dir_pth['ivscc_exc_RawFeatureWide']) 
    fmost_mf = pd.read_csv(dir_pth['fmost_exc_RawFeatureWide'])  
    print("ivscc_inh:", ivscc_inh.shape)
    print("ivscc_exc:", ivscc_exc.shape)
    print("fmost:", fmost_mf.shape)


    arbor_ids = [mystr.rstrip() for mystr in m_input['specimen_id']]
    print("Number of cells in arbor density file:", len(arbor_ids))

    print("...................................................")
    print(len([i for i in specimen_ids if i not in arbor_ids]), "cells do not have arbor density data!")

    print("...................................................")
    print("keeping only ids that are inside specimen id list")
    mask_arbor_ids = [True if i in specimen_ids else False for i in arbor_ids]
    arbor_ids = [b for a, b in zip(mask_arbor_ids, arbor_ids) if a]
    print("In total remains this amount of cells:", sum(mask_arbor_ids))

    print("...................................................")
    print("masking each arbor density channel and putting each channel in one specific key of a dict")
    soma_depth = np.squeeze(m_input['soma_depth'])[mask_arbor_ids]
    arbor_density = {}
    arbor_density['ax'] = m_input['hist_ax_de_api_bas'][:,:,:,0][mask_arbor_ids]
    arbor_density['de'] = m_input['hist_ax_de_api_bas'][:,:,:,1][mask_arbor_ids]
    arbor_density['api'] = m_input['hist_ax_de_api_bas'][:,:,:,2][mask_arbor_ids]
    arbor_density['bas'] = m_input['hist_ax_de_api_bas'][:,:,:,3][mask_arbor_ids]

    print("...................................................")
    print("create a 1d mask for each arbor density channel")
    # in the arbor density file, if a cell does not have M data, all the channels are nan values
    # if a cell is exc, the ax and de (first two channels) are all zeros and the api and bas (second 
    # two channels) have values. This mask is to find the exc cells with M data and inh cells with M data. 
    # so if all channels are nan, then it will return False. 

    not_valid_xm_channels = np.apply_over_axes(np.all, 
                                                np.logical_or(m_input['hist_ax_de_api_bas'] == 0, 
                                                np.isnan(m_input['hist_ax_de_api_bas'])),
                                                axes = [1, 2])
    not_valid_xm_channels = np.squeeze(not_valid_xm_channels)
    valid={}
    valid['ax'] = ~not_valid_xm_channels[:,0]
    valid['de']= ~not_valid_xm_channels[:,1]
    valid['api'] = ~not_valid_xm_channels[:,2]
    valid['bas']= ~not_valid_xm_channels[:,3]

    # Now for each channel, we mask for the valid cells and then we look if there is any
    # nan value in the cells arbor densities. If there is any nan, we should either 
    # remove that cell or put that nan to zero for that channel
    keep_cells = {}
    for k in arbor_density.keys():
        # mask for the cells with arbor density
        arbor_density[k] = arbor_density[k][valid[k]]
        # Find the m_cells with that have nan values in their non-OFF channels
        keep_cells[k] = ~np.squeeze(np.apply_over_axes(np.any, np.isnan(arbor_density[k]),axes = [1, 2]))
        # remove the cells that have nan values in each channel
        arbor_density[k] = arbor_density[k][keep_cells[k]]
        # collapse the last to dim and have a 1d arbor density for each channel
        arbor_density[k] = arbor_density[k].reshape(arbor_density[k].shape[0], -1)


    print("...................................................")
    print("PCA analysis ")
    n_comp_at_thr={}
    for k in arbor_density.keys():
        n_comp_at_thr[k] = proc_utils.get_PCA_explained_variance_ratio_at_thr(nparray=arbor_density[k], threshold=pca_th)
        if n_comp_at_thr[k] == 0:
            n_comp_at_thr[k] = 1
        # if n_comp_at_thr[k]> 20:
        #     n_comp_at_thr[k] = 20

        print(n_comp_at_thr[k])

    PC = {}
    for k in arbor_density.keys():
        if k != "ids":
            pca = PCA(n_comp_at_thr[k])
            # PC[k] = pca.fit_transform(arbor_density[k])[:, :20]
            PC[k] = pca.fit_transform(arbor_density[k])

            print(k, PC[k].shape)

    print("...................................................")
    print("Scaling PCA features")
    #Scaling PC features
    Scaled_PCs = {}
    total_var = pd.DataFrame(columns=PC.keys())

    for k in PC.keys():
        v = np.sqrt(np.sum(pd.DataFrame(PC[k]).var(axis=0)))
        Scaled_PCs[k] = PC[k] / v
        total_var.loc[0, k] = v

    # We save this cause we need them for the reconstruction of arbor densities
    total_var.to_csv(dir_pth['arbor_density_PC_vars_file'], index=False)

    print("...................................................")
    print("Removing outliers whithin 6 std from scaled PCs")
    #attaching specimen ids and removing outliers
    for k in PC.keys():
        Scaled_PCs[k] = pd.DataFrame(Scaled_PCs[k])
        scaling_thr = np.abs(np.max(Scaled_PCs[k].std(axis=0, skipna=True, numeric_only=True)) * 6)
        Scaled_PCs[k] = Scaled_PCs[k][(Scaled_PCs[k] < scaling_thr) & (Scaled_PCs[k] > -1 * scaling_thr)]
        Scaled_PCs[k].columns = [k + "_" + str(i) for i in range(Scaled_PCs[k].shape[1])]
        valid_ids = [id for i, id in enumerate(arbor_ids) if valid[k][i]]
        valid_ids = [id for i, id in enumerate(valid_ids) if keep_cells[k][i]]
        Scaled_PCs[k]["specimen_id"] = valid_ids
        Scaled_PCs[k]['specimen_id'] = Scaled_PCs[k]['specimen_id'].astype(str)

    #Merge all scaled PC features into one df
    data_frames = []
    for k in Scaled_PCs.keys():
        data_frames.append(Scaled_PCs[k])

    Scaled_PCs = reduce(lambda left, right: pd.merge(left, right, on=['specimen_id'], how='outer'), data_frames)

    df = Scaled_PCs.melt(value_vars=Scaled_PCs[[c for c in Scaled_PCs.columns if c != "specimen_id"]])
    sns.catplot(x="variable", y="value", kind='box', data=df, palette=sns.color_palette(["skyblue"]), aspect=4.4)
    ax = plt.gca()
    ax.set(**{'title': 'Scaled PC features', 'xlabel': '', 'ylabel': ''})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()


    # Plot m features
    # plots will not show in python
    fig, axs = plt.subplots(7,7, figsize=(30, 30))
    for ax, feature in zip(axs.flatten(), [i for i in ivscc_inh.columns if i!="specimen_id"]):
        ax.plot(ivscc_inh[feature], color='dodgerblue')
        ax.set_title(feature)

    fig, axs = plt.subplots(7,7, figsize=(30, 30))
    for ax, feature in zip(axs.flatten(), [i for i in ivscc_exc.columns if i!="specimen_id"]):
        ax.plot(ivscc_exc[feature], color='dodgerblue')
        ax.set_title(feature)

    fig, axs = plt.subplots(7,7, figsize=(30, 30))
    for ax, feature in zip(axs.flatten(), [i for i in fmost_mf.columns if i!="specimen_id"]):
        ax.plot(fmost_mf[feature], color='dodgerblue')
        ax.set_title(feature)

    print("..................................................")
    print("NOTE: Keep reliable features, Matt Malory provided this list")
    m_features = {"exc": ["specimen_id",
                      "apical_dendrite_max_euclidean_distance",
                      "apical_dendrite_bias_y",
                      "apical_dendrite_extent_y",
                      "apical_dendrite_soma_percentile_y",
                      "apical_dendrite_emd_with_basal_dendrite",
                      "apical_dendrite_frac_below_basal_dendrite",
                      "basal_dendrite_frac_above_apical_dendrite",
                      "apical_dendrite_depth_pc_0",
                      "apical_dendrite_total_length",
                      "basal_dendrite_bias_y",
                      "apical_dendrite_frac_above_basal_dendrite",
                      "basal_dendrite_soma_percentile_y",
                      "basal_dendrite_max_euclidean_distance",
                      "apical_dendrite_frac_intersect_basal_dendrite",
                      "basal_dendrite_extent_y",
                      "apical_dendrite_depth_pc_1",
                      "apical_dendrite_num_branches",
                      "apical_dendrite_extent_x",
                      "basal_dendrite_extent_x",
                      "basal_dendrite_total_length"],
                      
              "inh": ["specimen_id",
                      "basal_dendrite_max_euclidean_distance",
                      "basal_dendrite_bias_y",
                      "basal_dendrite_extent_y",
                      "basal_dendrite_soma_percentile_y",
                      "axon_emd_with_basal_dendrite",
                      "basal_dendrite_frac_below_axon",
                      "basal_dendrite_frac_above_axon",
                      "basal_dendrite_total_length",
                      "basal_dendrite_frac_intersect_axon",
                      "basal_dendrite_num_branches",
                      "basal_dendrite_extent_x"]}

    # Keep cells that are in locked dataset and only reliable features
    # Keep locked cells and reliable features from ivscc exc/inh and fmost data
    ivscc_exc['specimen_id'] = ivscc_exc['specimen_id'].astype(str)
    ivscc_inh['specimen_id'] = ivscc_inh['specimen_id'].astype(str)
    fmost_mf['specimen_id'] = fmost_mf['specimen_id'].astype(str)

    ivscc_exc = ivscc_exc[ivscc_exc['specimen_id'].isin(ids['specimen_id'].to_list())]
    ivscc_inh = ivscc_inh[ivscc_inh['specimen_id'].isin(ids['specimen_id'].to_list())]
    fmost_mf = fmost_mf[fmost_mf['specimen_id'].isin(ids['specimen_id'].to_list())]

    fmost_mf = fmost_mf[m_features['exc']]
    ivscc_exc = ivscc_exc[m_features['exc']]
    ivscc_inh = ivscc_inh[m_features['inh']]

    #Concat all the m features of all the ivscc and fmost cells together in one datafram
    # we will put nans for those that dont have features
    print("..................................................")
    print("Make all the features into one dataframe") 
    data_frames = [ivscc_exc, fmost_mf, ivscc_inh]
    m_features = reduce(lambda left, right: pd.merge(left, right, how='outer'), data_frames)
    m_features['specimen_id'] = m_features['specimen_id'].astype(str)

    print("size of the feature dataframe: ", m_features.shape )

    print("...................................................")
    print("Zscoring mfeatures features")
    subset_m_features = m_features[[c for c in m_features.columns if c != "specimen_id"]]
    m_features_norm = (subset_m_features - subset_m_features.mean(axis=0)) / subset_m_features.std(axis=0)

    print("Removing extreme mfeatures values (within 6 std)")
    scaling_thr = m_features_norm.std(axis=0, skipna=True, numeric_only=True) * 6
    m_features_norm = m_features_norm.reset_index()
    df1 = pd.melt(m_features_norm, id_vars=['index'], value_vars=[c for c in m_features_norm if c != "index"])
    df2 = pd.DataFrame(scaling_thr).reset_index().rename(columns={"index": 'variable', 0:"thr_std"})
    df3 = df1.merge(df2, on="variable")
    df3['new_value'] = np.where((df3['value'] < df3['thr_std']) & (df3['value'] > -1 * df3['thr_std'])
                            , df3['value'], np.nan)
    m_features_norm = df3.pivot(index='index', columns="variable", values="new_value")
    m_features_norm['specimen_id'] = m_features['specimen_id'].astype(str)

    df = m_features_norm.melt(value_vars=m_features_norm[[c for c in m_features_norm.columns if c != "specimen_id"]])
    sns.catplot(x="variable", y="value", kind='box', data=df, palette=sns.color_palette(["coral"]),aspect=2.4)
    ax = plt.gca()
    ax.set(**{'title': 'mfeatures features', 'xlabel': '', 'ylabel':''})
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.show()

    print("Scaled PCs size:", Scaled_PCs.shape)
    print("...................................................")
    print("adding soma depth and other m features to M data")
    m_cells = ~np.isnan(soma_depth)
    m_cells = [id for i, id in enumerate(arbor_ids) if m_cells[i]]

    sd = pd.DataFrame({"specimen_id": m_cells, "soma_depth": soma_depth[~np.isnan(soma_depth)]})

    data_frames = [Scaled_PCs, sd, m_features_norm]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['specimen_id'], how='inner'), data_frames)
    df_merged = df_merged.merge(pd.DataFrame(specimen_ids, columns=["specimen_id"]), on="specimen_id", how='right')


    # Make sure the order is the same as the locked id spec_ids
    df_merged = df_merged.set_index('specimen_id')
    df_merged = df_merged.loc[specimen_ids].reset_index()
    df_merged.to_csv(dir_pth['arbor_density_PC_and_features_file'], index=False)

    f = df_merged
    df = f.melt(value_vars=f[[c for c in f.columns if c != "specimen_id"]])
    sns.catplot(x="variable", y="value", kind='box', data=df, palette=sns.color_palette(["skyblue"]), aspect=4.4)
    ax = plt.gca()
    ax.set(**{'title': 'Scaled sPC features', 'xlabel': '', 'ylabel': ''})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()

    print("Size of Merged PCs features:", df_merged.shape)
    print("Done")
    
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))