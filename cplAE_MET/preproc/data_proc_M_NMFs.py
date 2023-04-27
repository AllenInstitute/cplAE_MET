
# This file is to process all the arbor densities and 
# generate their non-ngative matrix factorization

# %%
# import from python
import csv
import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce

# import from cplAE_MET
from cplAE_MET.utils.load_config import load_config
from cplAE_MET.utils.analysis_helpers import get_NMF_explained_variance_ratio


config_file = 'config_preproc.toml'
nmf_comp_th = 0.99

# %%
def set_paths(config_file=None):
    paths = load_config(config_file=config_file, verbose=False)
    paths['input'] = f'{str(paths["data_dir"])}'
    paths['arbor_density_file'] = f'{paths["input"]}/{str(paths["arbor_density_file"])}'
    paths['m_output_file'] = f'{paths["input"]}/{str(paths["m_output_file"])}'
    paths['specimen_ids'] = f'{paths["input"]}/{str(paths["specimen_ids_file"])}'
    return paths

def load_input_m_files(dir_pth):
    # Read locked dataset spec ids ##########
    ids = pd.read_csv(dir_pth['specimen_ids'])
    specimen_ids = [str(i) for i in ids['specimen_id'].tolist()]
    print("There are", len(specimen_ids), "sample_ids in the locked dataset")
    # Read arbor densities that Olga calculated ##########
    print("Loading M arbor_densities")
    arbor_dens = sio.loadmat(dir_pth['arbor_density_file'])
    return specimen_ids, arbor_dens

def convert_spec_id_to_str(list_spec_ids):
    return [str(i).rstrip() for i in list_spec_ids]

# %%
dir_pth = set_paths(config_file='config_preproc.toml')
locked_specimen_ids, arbor_dens  = load_input_m_files(dir_pth)

# Check the list of spec ids in the locked dataset and in the arbor density file ##########
arbor_ids = convert_spec_id_to_str(arbor_dens['specimen_id'])
#keeping only ids that are inside locked specimen id list")
mask_arbor_ids = [True if i in locked_specimen_ids else False for i in arbor_ids]
arbor_ids = [b for a, b in zip(mask_arbor_ids, arbor_ids) if a]

print("...................................................")
print("masking each arbor density channel and putting each channel in one specific key of a dict")
soma_depth = np.squeeze(arbor_dens['soma_depth'])[mask_arbor_ids]
arbor_density = {}
for col, key in enumerate(['ax', 'de', 'api', 'bas']):
    arbor_density[key] = arbor_dens['hist_ax_de_api_bas'][:,:,:,col][mask_arbor_ids]

print("...................................................")
print("create a 1d mask for each arbor density channel")
# For the exc or inh cells, two channels have values and two other channels are set to zero. 
# For other cells that do not have any arbor densitiy available, all channels are nan. 
# Now we want to create a mask for each channel for the exc and inh cells. We are going to use
# them for the PCA calculations and the channels that we manually set to zero should not be included
# in the PCA calcualtions
not_valid_xm_channels = np.apply_over_axes(np.all, 
                                            np.logical_or(arbor_dens['hist_ax_de_api_bas'] == 0, 
                                            np.isnan(arbor_dens['hist_ax_de_api_bas'])),
                                            axes = [1, 2])
not_valid_xm_channels = np.squeeze(not_valid_xm_channels)
valid={}
for col, key in enumerate(['ax', 'de', 'api', 'bas']):
    valid[key] = ~not_valid_xm_channels[:,col]

print("Number of inh cells with the arbor density:", valid['ax'].sum())
print("Number of exc cells with the arbor density:", valid['api'].sum())
assert (np.all(np.where(valid['ax'])[0]==np.where(valid['de'])[0]))

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
# concat nmfs for ax and de together and for api and bas together
arbor_density['inh'] = np.concatenate((arbor_density['ax'], arbor_density['de']), axis=1)
arbor_density['exc'] = np.concatenate((arbor_density['api'], arbor_density['bas']), axis=1)
keep_cells['inh'] = keep_cells['ax']
keep_cells['exc'] = keep_cells['api']
arbor_density.pop("ax")
arbor_density.pop("de")
arbor_density.pop("api")
arbor_density.pop("bas")

# %%
assert (np.all(np.where(valid['ax'])[0]==np.where(valid['de'])[0]))
assert (np.all(np.where(valid['api'])[0]==np.where(valid['bas'])[0]))
valid['inh'] = valid['ax']
valid['exc'] = valid['api']
valid.pop('ax')
valid.pop('de')
valid.pop('api')
valid.pop('bas')

# %% 
keep_cells['exc'] = keep_cells['api']
keep_cells['inh'] = keep_cells['ax']
keep_cells.pop('ax')
keep_cells.pop('de')
keep_cells.pop('api')
keep_cells.pop('bas')


# %%
# Run NMF and keep as many as components that is necessary keep the reconstuction error less than the threshold
NMF_comps = {}
NMF_comps['inh'] = 200
NMF_comps['exc'] = 200

# %%
# Compute nmf trnsforms and save the components for later reconstructions in the model
NMF_components = {}
NMF_transforms = {}
for k, v in arbor_density.items():
    model = NMF(n_components=NMF_comps[k], init='random', random_state=0, max_iter=200)
    W = model.fit_transform(v)
    H = model.components_
    NMF_components[k] = H
    NMF_transforms[k] = W
    print(k, NMF_transforms[k].shape) 


# %% 
# Scale the nmfs
Scaled_NMF = {}
total_var = pd.DataFrame(columns=NMF_transforms.keys())

for k in NMF_transforms.keys():
    v = np.sqrt(np.sum(pd.DataFrame(NMF_transforms[k]).var(axis=0)))
    print("total var along:", k, "is:", v)
    Scaled_NMF[k] = NMF_transforms[k] / v
    total_var.loc[0, k] = v


# %% 
# plot nmf transformations to see if we need scaling
for k in Scaled_NMF.keys():
    Scaled_NMF[k] = pd.DataFrame(Scaled_NMF[k])
    scaling_thr = np.abs(np.max(Scaled_NMF[k].std(axis=0, skipna=True, numeric_only=True)) * 6)
    Scaled_NMF[k] = Scaled_NMF[k][(Scaled_NMF[k] < scaling_thr) & (Scaled_NMF[k] > -1 * scaling_thr)]
    Scaled_NMF[k].columns = [k + "_" + str(i) for i in range(Scaled_NMF[k].shape[1])]
    valid_ids = [id for i, id in enumerate(arbor_ids) if valid[k][i]]
    valid_ids = [id for i, id in enumerate(valid_ids) if keep_cells[k][i]]
    Scaled_NMF[k]["specimen_id"] = valid_ids
    Scaled_NMF[k]['specimen_id'] = Scaled_NMF[k]['specimen_id'].astype(str)

# %%
data_frames = []
for k in Scaled_NMF.keys():
    data_frames.append(Scaled_NMF[k])
Scaled_NMF = reduce(lambda left, right: pd.merge(left, right, on=['specimen_id'], how='outer'), data_frames)
df = Scaled_NMF.melt(value_vars=Scaled_NMF[[c for c in Scaled_NMF.columns if c != "specimen_id"]])
sns.catplot(x="variable", y="value", kind='box', data=df, palette=sns.color_palette(["skyblue"]), aspect=4.4)
ax = plt.gca()
ax.set(**{'title': 'Scaled NMF features', 'xlabel': '', 'ylabel': ''})
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()

# %%
print("Scaled NMFs size:", Scaled_NMF.shape)
# adding soma depth 
m_cells = ~np.isnan(soma_depth)
m_cells = [id for i, id in enumerate(arbor_ids) if m_cells[i]]

# sd = pd.DataFrame({"specimen_id": m_cells, "soma_depth": soma_depth[~np.isnan(soma_depth)]})
print("Super important to set the nan values for the M cells PCs equal to zero before combining all other cells to the M cells")
print("this is because later on, we will mask only the m cells for the loss function and in there, we need to compute the loss")
print("on all the 4 channels. becasue if we dont do this then we will not include all the 4 channels in the loss calculations")
Scaled_NMF = Scaled_NMF.fillna(0)

# data_frames = [Scaled_NMF, sd]
# df_merged = reduce(lambda left, right: pd.merge(left, right, on=['specimen_id'], how='inner'), data_frames)

df_merged = Scaled_NMF
df_merged = df_merged.merge(pd.DataFrame(locked_specimen_ids, columns=["specimen_id"]), on="specimen_id", how='right')

# Make sure the order is the same as the locked id spec_ids
df_merged = df_merged.set_index('specimen_id')
df_merged = df_merged.loc[locked_specimen_ids].reset_index()
# df_merged.to_csv(dir_pth['arbor_density_PC_file'], index=False)

f = df_merged
df = f.melt(value_vars=f[[c for c in f.columns if c != "specimen_id"]])
sns.catplot(x="variable", y="value", kind='box', data=df, palette=sns.color_palette(["skyblue"]), aspect=4.4)
ax = plt.gca()
ax.set(**{'title': 'Scaled sPC features', 'xlabel': '', 'ylabel': ''})
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()

print("Size of Merged PCs features:", df_merged.shape)

# %%
# from cplAE_MET.utils.plots import plot_m

# m_nmfs = np.array(df_merged.drop(columns=["specimen_id"]))

# min_lim = 0
# rec_channel = {}
# for channel in ['inh', 'exc']:
#     col_limit = (min_lim , min_lim + NMF_components[channel].shape[0])
#     rec_channel[channel] = (np.dot(m_nmfs[:, col_limit[0]:col_limit[1]] * total_var[channel][0], NMF_components[channel]))
#     min_lim = col_limit[1]

# ax = rec_channel['inh'][:,0:480].reshape(-1, 120, 4)
# de = rec_channel['inh'][:,480:].reshape(-1, 120, 4)
# api = rec_channel['exc'][:,0:480].reshape(-1, 120, 4)
# bas = rec_channel['exc'][:,480:].reshape(-1, 120, 4)

# rec_arbor_density = np.stack((ax, de, api, bas ), axis=3) 

# %%
sio.savemat(dir_pth['m_output_file'], {'hist_ax_de_api_bas': arbor_dens['hist_ax_de_api_bas'],
                                       'm_features': np.array(df_merged.drop(columns=["specimen_id"])),
                                       'm_features_names': np.array(df_merged.drop(columns=["specimen_id"]).columns.to_list()),
                                    #    'soma_depth': arbor_dens['soma_depth'][0],
                                       'specimen_id': df_merged['specimen_id'].to_list(),
                                       'nmf_components_inh': NMF_components['inh'],
                                       'nmf_components_exc': NMF_components['exc'],
                                       'nmf_total_vars_inh': total_var['inh'][0],
                                       'nmf_total_vars_exc': total_var['exc'][0]}, do_compression=True)
print("Done")
# %%
