#########################################################
########### Preprocessing M, E and T data ###############
#########################################################
import json
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from functools import reduce
import cplAE_MET.utils.analysis_helpers as analysis

#Read the json file with all input args
# parser = argparse.ArgumentParser()
# parser.add_argument("-f", "--input", required=True, type=str)
# args = parser.parse_args()
#
# with open(args.input) as json_data:
#     data = json.load(json_data)

data = {"input_path": "/Users/fahimehb/Documents/git-workspace/cplAE_MET/data/proc/",
 "output_path": "/Users/fahimehb/Documents/git-workspace/cplAE_MET/data/proc/",
 "T_data_file": "data.feather",
 "T_annotation_file": "anno.feather",
 "gene_file": "good_genes_beta_score.csv",
 "specimen_ids_file": "revised_manuscript_specimen_ids.txt",
 "E_timeseries_file": "inh_fv_Ephys_run_all_20July2021.h5",
 "ipfx_features_file": "inh_ipfx_features_20July2021.csv",
 "M_output_data_file": "M_data_25Aug2021.mat",
 "M_output_anno_file": "M_anno_25Aug2021.csv",
 "M_input_anno_file": "manual_auto823.csv",
 "M_autotrace_anno_file":  "auto280.csv",
 "beta_threshold": 0.4,
 "pca_comp_threshold": 0.97,
 "output_file_prefix": "inh",
 "hist2d_120x4_folder": "hist2d_120x4"
}


#Input vars
output_file_prefix = data['output_file_prefix']
T_data_path = data['input_path'] + "/" + output_file_prefix + "_T_data.csv"
T_anno_path = data['input_path'] + "/" + output_file_prefix + "_T_annotations.csv"
E_data_path = data['input_path'] + "/" + output_file_prefix + "_Merged_Ephys_features.csv"
M_data_path = data['input_path'] + "/" + data["M_output_data_file"]
M_anno_path = data['input_path'] + "/" + data["M_output_anno_file"]
gene_file_path = data['input_path'] + data['gene_file']
specimen_path = data['input_path'] + data['specimen_ids_file']
output_path = data['output_path']

print("...................................................")
print("Loading E, T and M data")
E_data = pd.read_csv(E_data_path)
print("shape of E data:", E_data.shape)
T_data = pd.read_csv(T_data_path)
print("shape of T data:", T_data.shape)
M_dat = sio.loadmat(M_data_path)
print("shape of hist_ax_de data:", M_dat['hist_ax_de'].shape)
T_annotation = pd.read_csv(T_anno_path)
print("length of T_annotation:", len(T_annotation))
M_annotation = pd.read_csv(M_anno_path)
print("length of M_annotation:", len(M_annotation))


print("...................................................")
print("Combining M, E and T data")
#Align E and T with M_anno file 
#This is becasue M data mat does not have sample id info so we assume M_anno sample_id is the correct order
ME = reduce(lambda  left,right: pd.merge(left, right, on=['sample_id'], how='left'), 
             [M_annotation[["sample_id"]], 
              E_data])

MT = reduce(lambda  left,right: pd.merge(left, right, on=['sample_id'], how='left'), 
             [M_annotation[["sample_id"]],
              T_data])

# Merge with T_annotation
MET = reduce(lambda  left,right: pd.merge(left, right, on=['sample_id'], how='left'), 
             [ME, MT, T_annotation[["sample_id", "Tree_first_cl_id", "Tree_first_cl_label", "Tree_first_cl_color"]]])

print("...................................................")
print("Writing the output mat")

#writing M, E and T data
model_input_mat = {}
model_input_mat["E_dat"] = np.array(MET[[c for c in E_data.columns if c!="sample_id"]])
model_input_mat["T_dat"] = np.array(MET[[c for c in T_data.columns if c!="sample_id"]])
model_input_mat["M_dat"]= np.array(M_dat["hist_ax_de"])
model_input_mat["soma_depth"] = np.array(M_dat["soma_depth"])

#masking M, E and T data for the nans
# mask_T = ~np.isnan(model_input_mat["T_dat"])
# mask_E = ~np.isnan(model_input_mat["E_dat"])
# mask_M = ~np.isnan(model_input_mat["M_dat"])
# mask_soma_depth = ~np.isnan(model_input_mat["soma_depth"])
_, mask_M = analysis.remove_nan_observations(model_input_mat["M_dat"])
_, mask_E = analysis.remove_nan_observations(model_input_mat["E_dat"])
_, mask_T = analysis.remove_nan_observations(model_input_mat["T_dat"])
_, mask_soma_depth = analysis.remove_nan_observations(model_input_mat["soma_depth"][0])

#Make sure that M mask and soma_depth mask are equal
# assert np.all(np.equal(mask_soma_depth, mask_M))

#writing the sample_ids and the masks and some meta data
model_input_mat["sample_id"] = np.array(MET['sample_id'])

model_input_mat["mask_M"] = np.array(mask_M)
model_input_mat["mask_E"] = np.array(mask_E)
model_input_mat["mask_T"] = np.array(mask_T)
model_input_mat["mask_soma_depth"] = np.array(mask_soma_depth)

model_input_mat["cluster_id"] = np.array(MET.Tree_first_cl_id)
model_input_mat["cluster_color"] = np.array(MET.Tree_first_cl_color)
model_input_mat["cluster"] = np.array(MET.Tree_first_cl_label)

#Saving input mat
print("Size of M data:", model_input_mat['M_dat'].shape)
print("Size of M data:", model_input_mat["soma_depth"].shape)
print("Size of E data:", model_input_mat["E_dat"].shape)
print("Size of T data:", model_input_mat["T_dat"].shape)
print("saving!")
sio.savemat(output_path + "/" + output_file_prefix + "_MET_model_input_mat.mat", model_input_mat)
