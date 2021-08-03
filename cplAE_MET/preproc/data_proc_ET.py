#########################################################
############ Preprocessing T and E data #################
#########################################################
import csv
import h5py
import json
import feather
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from cplAE_TE.utils.load_helpers import get_paths, load_dataset, load_summary_files


#Read the json file with all input args
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--input", required=True, type=str)
args = parser.parse_args()

with open(args.input) as json_data:
    data = json.load(json_data)

#Input vars
output_file_prefix = data['output_file_prefix']
T_data_path = data['input_path'] + "/" + output_file_prefix + "_T_data.csv"
T_anno_path = data['input_path'] + "/" + output_file_prefix + "_T_annotations.csv"
E_data_path = data['input_path'] + "/" + output_file_prefix + "_Merged_Ephys_features.csv"
gene_file_path = data['input_path'] + data['gene_file']
specimen_path = data['input_path'] + data['specimen_ids_file']
output_path = data['output_path']


print("...................................................")
print("Loading E  and T data")
E_data = pd.read_csv(E_data_path)
T_data = pd.read_csv(T_data_path)
annotation = pd.read_csv(T_anno_path)

# Put all of them in the same order by sample_id
all_merged = annotation.merge(E_data.merge(T_data, on="sample_id"), on="sample_id")
print(all_merged.shape)

#separate them
E_data = all_merged[E_data.columns]
T_data = all_merged[T_data.columns]
annotation = all_merged[annotation.columns]

model_input_mat = {}
model_input_mat["E_dat"] = np.array(E_data.drop("sample_id", axis=1))
model_input_mat["T_dat"] = np.array(T_data.drop("sample_id", axis=1))
model_input_mat["sample_id"] = np.array(all_merged.sample_id)
model_input_mat["cluster_id"] = np.array(all_merged.Tree_first_cl_id)
model_input_mat["cluster_color"] = np.array(all_merged.Tree_first_cl_color)
model_input_mat["cluster"] = np.array(all_merged.Tree_first_cl_label)

f = E_data
df = f.melt(value_vars=f[[c for c in f.columns if c != "sample_id"]])
sns.catplot(x="variable", y="value",kind='box', data=df, palette=sns.color_palette(["skyblue"]),aspect=4.4)
ax = plt.gca()
ax.set(**{'title':'Scaled sPC features','xlabel':'','ylabel':''})
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()

print("Size of T data:", model_input_mat['T_dat'].shape, "Size of E data:", model_input_mat["E_dat"].shape)
print("saving!")
sio.savemat(output_path + "/" + output_file_prefix + "_model_input_mat.mat", model_input_mat)
print("Done")
