# %% [markdown]
# ### MET triple modality autoencoder

# %%
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from functools import reduce
from cplAE_MET.utils.load_config import load_config
import cplAE_MET.utils.analysis_helpers as proc_utils


config_file = 'config_preproc.toml'
pca_th=0.90

# %%
def set_paths(config_file=None):
    paths = load_config(config_file=config_file, verbose=False)
    paths['input'] = f'{str(paths["data_dir"])}'
    paths['arbor_density_file'] = f'{paths["input"]}/{str(paths["arbor_density_file"])}'
    # paths['ivscc_inh_m_features'] = f'{paths["input"]}/{str(paths["ivscc_inh_m_features"])}'
    # paths['ivscc_exc_m_features'] = f'{paths["input"]}/{str(paths["ivscc_exc_m_features"])}'
    # paths['fmost_exc_m_features'] = f'{paths["input"]}/{str(paths["fmost_exc_m_features"])}'
    paths['me_m_features'] = f'{paths["input"]}/{str(paths["me_m_features"])}'
    paths['em_m_features'] = f'{paths["input"]}/{str(paths["em_m_features"])}' ""
    paths['m_output_file'] = f'{paths["input"]}/{str(paths["m_output_file"])}'
    paths['specimen_ids'] = f'{paths["input"]}/{str(paths["specimen_ids_file"])}'
    return paths


def get_m_features():
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
    return m_features

def convert_spec_id_to_str(list_spec_ids):
    return [str(i).rstrip() for i in list_spec_ids]

def load_input_m_files(dir_pth):
    # Read locked dataset spec ids ##########
    ids = pd.read_csv(dir_pth['specimen_ids'])
    specimen_ids = [str(i) for i in ids['specimen_id'].tolist()]
    print("There are", len(specimen_ids), "sample_ids in the locked dataset")
    # Read arbor densities that Olga calculated ##########
    print("Loading M arbor_densities")
    m_input = sio.loadmat(dir_pth['arbor_density_file'])
    # read morpho-metric_features 
    ivscc_inh = pd.read_csv(dir_pth['ivscc_inh_m_features'])
    ivscc_exc = pd.read_csv(dir_pth['ivscc_exc_m_features']) 
    fmost_mf = pd.read_csv(dir_pth['fmost_exc_m_features']) 
    me_mf = pd.read_csv(dir_pth['me_m_features']) 
    em_mf = pd.read_csv(dir_pth['em_m_features']) 
    ivscc_exc['specimen_id'] = ivscc_exc['specimen_id'].astype(str)
    ivscc_inh['specimen_id'] = ivscc_inh['specimen_id'].astype(str)
    fmost_mf['specimen_id'] = fmost_mf['specimen_id'].astype(str)
    me_mf['specimen_id'] = me_mf['specimen_id'].astype(str)
    em_mf['specimen_id'] = em_mf['specimen_id'].astype(str) 
    print("size of ivscc_exc, inh fmost, em and me morphometric features")
    print(ivscc_exc.shape, ivscc_inh.shape, fmost_mf.shape, me_mf.shape, em_mf.shape)
    return specimen_ids, m_input, ivscc_inh, ivscc_exc, fmost_mf, me_mf, em_mf


######################################################
dir_pth = set_paths(config_file='config_preproc.toml')
mf_dataset = {}
locked_specimen_ids, m_input, mf_dataset['ivscc_inh'], mf_dataset['ivscc_exc'], mf_dataset['fmost_mf'], mf_dataset['me_mf'], mf_dataset['em_mf'] = load_input_m_files(dir_pth)

# Check the list of spec ids in the locked dataset and in the arbor density file ##########
arbor_ids = convert_spec_id_to_str(m_input['specimen_id'])
print("Number of cells in arbor density file:", len(arbor_ids))
print(len([i for i in locked_specimen_ids if i not in arbor_ids]), "cells do not have arbor density data!")


print("...................................................")
print("keeping only ids that are inside locked specimen id list")
mask_arbor_ids = [True if i in locked_specimen_ids else False for i in arbor_ids]
arbor_ids = [b for a, b in zip(mask_arbor_ids, arbor_ids) if a]
print("In total remains this amount of cells:", sum(mask_arbor_ids))

print("...................................................")
print("masking each arbor density channel and putting each channel in one specific key of a dict")
soma_depth = np.squeeze(m_input['soma_depth'])[mask_arbor_ids]
arbor_density = {}
for col, key in enumerate(['ax', 'de', 'api', 'bas']):
    arbor_density[key] = m_input['hist_ax_de_api_bas'][:,:,:,col][mask_arbor_ids]

print("...................................................")
print("create a 1d mask for each arbor density channel")
# For the exc or inh cells, two channels have values and two other channels are set to zero. 
# For other cells that do not have any arbor densitiy available, all channels are nan. 
# Now we want to create a mask for each channel for the exc and inh cells. We are going to use
# them for the PCA calculations and the channels that we manually set to zero should not be included
# in the PCA calcualtions
not_valid_xm_channels = np.apply_over_axes(np.all, 
                                            np.logical_or(m_input['hist_ax_de_api_bas'] == 0, 
                                            np.isnan(m_input['hist_ax_de_api_bas'])),
                                            axes = [1, 2])
not_valid_xm_channels = np.squeeze(not_valid_xm_channels)
valid={}
for col, key in enumerate(['ax', 'de', 'api', 'bas']):
    valid[key] = ~not_valid_xm_channels[:,col]

print("Number of inh cells with the arbor density:", valid['ax'].sum())
print("Number of exc cells with the arbor density:", valid['api'].sum())

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
    # collapse the last two dims and have a 1d arbor density for each channel
    arbor_density[k] = arbor_density[k].reshape(arbor_density[k].shape[0], -1)
    print("Number of cells in this channel for PCA calculations. The rest are removed if they had nans:", k, keep_cells[k].sum())

# %%
# Run PCA and keep as many as components that is necessary to explain 97% of the variance in the data
print("...................................................")
print("PCA analysis ")
PC = {}
PC_means = {}
PC_comps = {}
for k in arbor_density.keys():
    if k != "ids":
        n_comp = proc_utils.get_PCA_explained_variance_ratio_at_thr(nparray=arbor_density[k], threshold=pca_th)
        if n_comp == 0:
            n_comp = 1
        pca = PCA(n_comp)
        pca.fit(arbor_density[k])
        PC[k] = np.dot(arbor_density[k] - pca.mean_, pca.components_.T)
        PC_means[k] = pca.mean_
        PC_comps[k] = pca.components_
        print(k, PC[k].shape)


# %%
print("...................................................")
print("Scaling PCA features with the TOTAL var in each channel")
#Scaling PC features
Scaled_PCs = {}
total_var = pd.DataFrame(columns=PC.keys())

for k in PC.keys():
    v = np.sqrt(np.sum(pd.DataFrame(PC[k]).var(axis=0)))
    Scaled_PCs[k] = PC[k] / v
    total_var.loc[0, k] = v



# %%
print("...................................................")
print("Putting all ScaledPCs into a dataframe")
for k in PC.keys():
    Scaled_PCs[k] = pd.DataFrame(Scaled_PCs[k])
    # scaling_thr = np.abs(np.max(Scaled_PCs[k].std(axis=0, skipna=True, numeric_only=True)) * 6)
    # Scaled_PCs[k] = Scaled_PCs[k][(Scaled_PCs[k] < scaling_thr) & (Scaled_PCs[k] > -1 * scaling_thr)]
    Scaled_PCs[k].columns = [k + "_" + str(i) for i in range(Scaled_PCs[k].shape[1])]
    valid_ids = [id for i, id in enumerate(arbor_ids) if valid[k][i]]
    valid_ids = [id for i, id in enumerate(valid_ids) if keep_cells[k][i]]
    Scaled_PCs[k]["specimen_id"] = valid_ids
    Scaled_PCs[k]['specimen_id'] = Scaled_PCs[k]['specimen_id'].astype(str)

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


# If you like to plot the morpho_metric features
# fig, axs = plt.subplots(7,7, figsize=(30, 30))
# for ax, feature in zip(axs.flatten(), [i for i in ivscc_exc.columns if i!="specimen_id"]):
#     ax.plot(ivscc_exc[feature], color='dodgerblue')
#     ax.set_title(feature)

# %%
print("...................................................")
print("Size of the morphometric features for ivscc_exc, ivscc_inh, fmost, me and em cells")
print(mf_dataset['ivscc_exc'].shape, mf_dataset['ivscc_inh'].shape, \
      mf_dataset['fmost_mf'].shape, mf_dataset['me_mf'].shape, mf_dataset['em_mf'].shape)

print("")
print("Size when we remove all the cells that are not in the locked dataset")
for k in mf_dataset.keys():
    mf_dataset[k] = mf_dataset[k][mf_dataset[k]['specimen_id'].isin(locked_specimen_ids)]
print(mf_dataset['ivscc_exc'].shape, mf_dataset['ivscc_inh'].shape, \
      mf_dataset['fmost_mf'].shape, mf_dataset['me_mf'].shape, mf_dataset['em_mf'].shape)

# %%
# exc_cells = mf_dataset['ivscc_exc']['specimen_id'].to_list() + \
            

#%%
m_features_cols = get_m_features()['exc']+ get_m_features()['inh']
m_features_cols = np.unique(m_features_cols)
print("")
print("Now we keep only exc and inh cols that are reliable")
for k in mf_dataset.keys():
    keep_cols = [col for col in mf_dataset[k].columns if col in m_features_cols]
    mf_dataset[k] = mf_dataset[k][keep_cols]
    new_cols = [col for col in m_features_cols if col not in mf_dataset[k].columns]
    mf_dataset[k][new_cols] = 0.
print(mf_dataset['ivscc_exc'].shape, mf_dataset['ivscc_inh'].shape, \
      mf_dataset['fmost_mf'].shape, mf_dataset['me_mf'].shape, mf_dataset['em_mf'].shape)

print("")
print("Rm the cells that have no arbor density but have m features")
m_cells_w_feature = []
for k in mf_dataset.keys():
    m_cells_w_feature.append([str(i) for i in mf_dataset[k].specimen_id.to_list()])
m_cells_w_feature = [item for sublist in m_cells_w_feature for item in sublist]

m_cells_w_arborPCs = [str(i) for i in Scaled_PCs['specimen_id'].to_list()]

print("m_cells with m features:", len(m_cells_w_feature))
print("m_cells with arbor PCs:", len(m_cells_w_arborPCs))
rm_cells = [i for i in m_cells_w_feature if i not in m_cells_w_arborPCs]
print("m cells with features that are NOT in m cells with arbor PCs:",len(rm_cells))
print("m cells with features that are NOT in locked dataset:", len([i for i in m_cells_w_feature if i not in locked_specimen_ids]))

# %%
print("")
print("so we should remove this cells from our analysis:")
for k in mf_dataset.keys():
    mf_dataset[k] = mf_dataset[k][mf_dataset[k]['specimen_id'].isin(m_cells_w_arborPCs)]
print(mf_dataset['ivscc_exc'].shape, mf_dataset['ivscc_inh'].shape, \
      mf_dataset['fmost_mf'].shape, mf_dataset['me_mf'].shape, mf_dataset['em_mf'].shape)

print("")
print("Put all m features into a dataframe and add specimen_ids")
data_frames = [v for _, v in mf_dataset.items()]
m_features = reduce(lambda left, right: pd.merge(left, right, how='outer'), data_frames)
m_features['specimen_id'] = m_features['specimen_id'].astype(str)

#%%
print("...................................................")
print("Zscoring mfeatures features")
subset_m_features = m_features[[c for c in m_features.columns if c != "specimen_id"]]
m_features_norm = (subset_m_features - subset_m_features.mean(axis=0)) / subset_m_features.std(axis=0)
df = m_features_norm.melt(value_vars=m_features_norm[[c for c in m_features_norm.columns if c != "specimen_id"]])
sns.catplot(x="variable", y="value", kind='box', data=df, palette=sns.color_palette(["coral"]),aspect=2.4)
ax = plt.gca()
ax.set(**{'title': 'mfeatures features', 'xlabel': '', 'ylabel':''})
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()

# %%
print("Removing extreme mfeatures values (within 6 std)")
scaling_thr = m_features_norm.std(axis=0, skipna=True, numeric_only=True) * 7
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

#%%
print("...................................................")
print("Putting scaled PCs, soma depth and m_features to one dataframe")
m_cells = ~np.isnan(soma_depth)
m_cells = [id for i, id in enumerate(arbor_ids) if m_cells[i]]
sd = pd.DataFrame({"specimen_id": m_cells, "soma_depth": soma_depth[~np.isnan(soma_depth)]})
print("Scaled PCs size:", Scaled_PCs.shape)
print("soma_depth size:", sd.shape)
print("mfeatures size:", m_features_norm.shape)
print("")
print("Super important to set the nan values for the M cells PCs equal to zero before combining all other cells to the M cells")
print("this is because later on, we will mask only the m cells for the loss function and in there, we need to compute the loss")
print("on all the 4 channels. becasue if we dont do this then we will not include all the 4 channels in the loss calculations")
Scaled_PCs = Scaled_PCs.fillna(0)
#%%
data_frames = [Scaled_PCs, sd, m_features_norm]
df_merged = reduce(lambda left, right: pd.merge(left, right, on=['specimen_id'], how='left'), data_frames)
##################### From here
print("...................................................")
print("Removing some of the features that more than %5 of data dont have that feature")
# this feature has a lot of nan, so we drop this
df_merged = df_merged.drop(columns=["basal_dendrite_total_length"])
df_merged = df_merged[~df_merged.isnull().any(axis=1)]
#################### Until here if you want to get read of a feature and all the nans
# broadcast to all spec ids
df_merged = df_merged.merge(pd.DataFrame(locked_specimen_ids, columns=["specimen_id"]), on="specimen_id", how='right')

print("...................................................")
temp = df_merged[[c for c in df_merged.columns if c!="specimen_id"]]
print("In total, this amount of cells either have arbor PCs or m features: ", len(temp.index[~temp.isnull().all(1)]))

# %%
# Make sure the order is the same as the locked id spec_ids
df_merged = df_merged.set_index('specimen_id')
df_merged = df_merged.loc[locked_specimen_ids].reset_index()

# plot
f = df_merged
df = f.melt(value_vars=f[[c for c in f.columns if c != "specimen_id"]])
sns.catplot(x="variable", y="value", kind='box', data=df, palette=sns.color_palette(["skyblue"]), aspect=4.4)
ax = plt.gca()
ax.set(**{'title': 'Scaled sPC features', 'xlabel': '', 'ylabel': ''})
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()

print("Size of Merged PCs features:", df_merged.shape)

sio.savemat(dir_pth['m_output_file'], {'hist_ax_de_api_bas': m_input['hist_ax_de_api_bas'],
                                       'm_pcs_features': np.array(df_merged.drop(columns=["specimen_id"])),
                                       'm_pc_features_names': np.array(df_merged.drop(columns=["specimen_id"]).columns.to_list()),
                                       'soma_depth': m_input['soma_depth'][0],
                                       'specimen_id': df_merged['specimen_id'].to_list(),
                                       'pca_components_ax': PC_comps['ax'],
                                       'pca_components_de': PC_comps['de'],
                                       'pca_components_api': PC_comps['api'],
                                       'pca_components_bas': PC_comps['bas'],
                                       'pca_means_ax': PC_means['ax'],
                                       'pca_means_de': PC_means['de'],
                                       'pca_means_api': PC_means['api'],
                                       'pca_means_bas': PC_means['bas'], 
                                       'pca_total_vars_ax': total_var['ax'][0],
                                       'pca_total_vars_de': total_var['de'][0],
                                       'pca_total_vars_api': total_var['api'][0],
                                       'pca_total_vars_bas': total_var['bas'][0]}, do_compression=True)

print("Done")