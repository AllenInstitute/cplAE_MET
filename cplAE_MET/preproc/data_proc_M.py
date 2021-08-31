#########################################################
################ Preprocessing M data ###################
#########################################################
import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from collections import Counter
import cplAE_MET.utils.utils as ut


#Read the json file with all input args
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--input", required=True, type=str)
args = parser.parse_args()

with open(args.input) as json_data:
    data = json.load(json_data)


### Reading 823 total morpho cells
#This was provied by Olga from the following path:
#"/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Olga/morphology/arbor_density_ae/aspiny082421"

#Input vars
m_anno_path = data["input_path"] + data['M_input_anno_file']
hist2d_120x4_folder = data["input_path"] + data['hist2d_120x4_folder']
output_M_mat = data['output_path'] + data['M_output_data_file']
M_autotrace_anno_path = data['input_path'] + data['M_autotrace_anno_file']
M_output_anno_file_path = data['input_path'] + data['M_output_anno_file']

m_anno = pd.read_csv(m_anno_path)
print("...................................................")
print("Number of cells in the M_annotation file", len(m_anno))


t_anno = pd.read_csv(data['input_path'] + "/inh_T_annotations.csv")
t_anno = t_anno[["spec_id_label", "sample_id", 'Tree_first_cl_label', 'Tree_first_cl_color', 'Tree_first_cl_id', 'Tree_call_label']]
t_anno = t_anno.drop_duplicates().reset_index(drop=True).rename(
    columns={"Tree_first_cl_color": "cluster_color",
             "Tree_first_cl_id": "cluster_id",
             "spec_id_label":"specimen_id"})
print("...................................................")
print("Number of cells in the locked dataset", len(t_anno))

t_anno = t_anno[t_anno['Tree_call_label']!="PoorQ"]
print("...................................................")
print("We are removing PoorQ cells in the M data and then we will use M data to filter E and T data")
print("Number of GOOD cells in the locked dataset", len(t_anno))
print(Counter(t_anno['Tree_call_label']))


new_cells = ([i for i in m_anno['specimen_id'].tolist() if i not in t_anno['specimen_id'].tolist()])
ut.write_list_to_csv("/Users/fahimehb/Documents/git-workspace/cplAE_MET/data/proc/new_inh_cells.csv",
                     new_cells)
print("...................................................")
print("Some cells exist in M_anno but not in the locked dataset, we print them in the follwoing file and remove them "
      "from our data:")
print("/Users/fahimehb/Documents/git-workspace/cplAE_MET/data/proc/new_inh_cells.csv")


m_anno = t_anno.merge(m_anno, on="specimen_id", how='left')
m_anno = m_anno.rename(columns={"VISp Tree Mapping":"cluster"})


print("...................................................")
print("Generating image for all the locke dataset, for those that we ont have M, we put nan")
hist_shape = (1, 1, 120, 4)
im_shape = (1, 2, 120, 4)
ax_list = []
de_list = []
ax_list_2 = []
de_list_2 = []
drop_spec_id = []
im = np.zeros((m_anno['specimen_id'].size, 2, 120, 4), dtype=float)
soma_depth = np.zeros((m_anno['specimen_id'].size,))
count = 0

for i, spec_id in tqdm(enumerate(m_anno['specimen_id'])):
    if os.path.isfile(hist2d_120x4_folder + f'/hist2d_120x4_axon_{spec_id}.csv'):

        im_ax = pd.read_csv(hist2d_120x4_folder + f'/hist2d_120x4_axon_{spec_id}.csv', header=None).values
        im_de = pd.read_csv(hist2d_120x4_folder + f'/hist2d_120x4_dendrite_{spec_id}.csv', header=None).values

        if len(np.where(im_ax.reshape(-1))[0]) < 5:
            print("There are", len(np.where(im_ax.reshape(-1))[0]), "nonzero pixels in axon of", spec_id)
            drop_spec_id.append(spec_id)
        if len(np.where(im_de.reshape(-1))[0]) < 5:
            print("There are", len(np.where(im_de.reshape(-1))[0]), "nonzero pixels in dendrite of", spec_id)
            drop_spec_id.append(spec_id)

        # Normalize
        ax_list.append(np.sum(im_ax))
        de_list.append(np.sum(im_de))
        im_ax = im_ax * 1e2 / np.sum(im_ax)
        im_de = im_de * 1e2 / np.sum(im_de)

        im[i, ...] = (np.concatenate([im_de.reshape(hist_shape), im_ax.reshape(hist_shape)], axis=1))
        soma_depth[i] = np.squeeze(m_anno.loc[m_anno['specimen_id'] == spec_id]['soma_depth'].values)
        count = count + 1
    else:
        im[i, ...] = np.full(im_shape, np.nan)
        soma_depth[i] = np.nan

print(f'found data for {count} morphologies')


output_M_mat = data['output_path'] + data['M_output_data_file']
sio.savemat(output_M_mat, {'hist_ax_de':im,'soma_depth':soma_depth}, do_compression=True)
print("Writing the output mat in the following path:")
print(output_M_mat)

m_ind = np.where(~np.isnan(soma_depth))[0]
m_avbl = m_anno.loc[m_ind] # annotations for cells with morphology data
a_df = pd.read_csv(M_autotrace_anno_path)
M_anno = m_anno[['specimen_id', 'sample_id', 'cluster', 'cluster_color', 'cluster_id', "Tree_call_label"]].copy()
M_anno['reconstruction'] = "none"
M_anno.loc[M_anno['specimen_id'].isin(m_avbl['specimen_id']),'reconstruction'] = "manual"
M_anno.loc[M_anno['specimen_id'].isin(a_df['specimen_id']),'reconstruction'] = "auto"
M_anno['soma_depth'] = soma_depth
M_anno.to_csv(M_output_anno_file_path, index=False)
print("size of M_anno:",len(M_anno))
print("Writing the M anno in the following path:")
print(M_output_anno_file_path)
print("Done!")
