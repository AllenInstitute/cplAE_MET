#########################################################
############ Preprocessing T and E data #################
#########################################################
import h5py
import json
import feather
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cplAE_MET.utils.analysis_helpers as proc_utils


#Read the json file with all input args
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--input", required=True, type=str)
args = parser.parse_args()

with open(args.input) as json_data:
    data = json.load(json_data)

#Input vars
E_timeseries_path = data['input_path'] + data['E_timeseries_file']
ipfx_file_path = data['input_path'] + data['ipfx_features_file']
output_path = data['output_path']
output_file_prefix = data['output_file_prefix']
pca_comp_threshold = data['pca_comp_threshold']
T_anno_path = data['input_path'] + data['T_annotation_file']
specimen_path = data['input_path'] + data['specimen_ids_file']

ids = pd.read_csv(specimen_path, header=None)
ids.rename(columns = {0:'specimen_id'}, inplace = True)
print("...................................................")
print("There are", ids.shape[0], "sample_ids in the locked dataset")


print("...................................................")
print("Loading E data")
hf = h5py.File(E_timeseries_path, 'r')

h5_ids = np.array(hf.get("ids"))
print("Number of cells in h5(time series file):", len(h5_ids))

print("...................................................")
print(len([i for i in ids['specimen_id'].tolist() if i not in h5_ids]), "cells do not have time series data!")

print("...................................................")
print("keeping only ids that are inside the lockdown dataset")
mask_h5_ids = [True if i in ids['specimen_id'].tolist() else False for i in h5_ids]
h5_ids = h5_ids[mask_h5_ids]
print("In total remains this amount of cells:", sum(mask_h5_ids))

#Read time series into a dictionary and masking them for only the ids that exist in the locked dataset
time_series = {}
for k in hf.keys():
    time_series[k] = np.array(hf.get(k))[mask_h5_ids]

print("...................................................")
print("removing nan values from individual experiments")
# Check if there is any nan values in any of the np arrays
expt_with_nans = []
for sub_expt in time_series.keys():
    if sub_expt != "ids":
        if np.isnan(time_series[sub_expt]).any():
            print("nan values were detected in this experiment:", sub_expt)
            expt_with_nans.append(sub_expt)

# Remove nan values from the timeseries with nan
dropped_cells = {}
for sub_expt in expt_with_nans:
    time_series[sub_expt], dropped_cells[sub_expt] = \
        proc_utils.drop_nan_rows_or_cols(time_series[sub_expt], 1)

print("...................................................")
print("PCA analysis")
# Apply PCA and keep the n components as computed below
number_of_components = {}
for k in hf.keys():
    if k not in ["ids"]:
        print(k)
        n_comp_at_thr = proc_utils.get_PCA_explained_variance_ratio_at_thr(
            nparray=time_series[k], threshold=pca_comp_threshold)
        #If number of components are zero, then use 1
        if n_comp_at_thr == 0:
            number_of_components[k] = 1
        else:
            number_of_components[k] = n_comp_at_thr

PC = {}
for k in hf.keys():
    if k != "ids":
        pca = PCA(number_of_components[k])
        PC[k] = pca.fit_transform(time_series[k])
        print(k, PC[k].shape)

print("...................................................")
print("Scaling PCA features")
#Scaling PC features
Scaled_PCs = {}
total_var = {}

for k in PC.keys():
    total_var[k] = np.sqrt(np.sum(pd.DataFrame(PC[k]).var(axis=0)))
    Scaled_PCs[k] = PC[k] / total_var[k]
    if (proc_utils.check_for_nan(Scaled_PCs[k])):
        Scaled_PCs[k], _ = proc_utils.drop_nan_rows_or_cols(Scaled_PCs[k], axis=1)

print("...................................................")
print("Removing outliers whithin 6 std from scaled PCs")
#attaching specimen ids and removing outliers
for k in PC.keys():
    Scaled_PCs[k] = pd.DataFrame(Scaled_PCs[k])
    scaling_thr = np.abs(np.max(Scaled_PCs[k].std(axis=0, skipna=True, numeric_only=True)) * 6)
    Scaled_PCs[k] = Scaled_PCs[k][(Scaled_PCs[k] < scaling_thr) & (Scaled_PCs[k] > -1 * scaling_thr)]
    Scaled_PCs[k].columns = [k + "_" + str(i) for i in range(Scaled_PCs[k].shape[1])]
    if k not in dropped_cells.keys():
        Scaled_PCs[k]["specimen_id"] = h5_ids
    else:
        Scaled_PCs[k]["specimen_id"] = np.delete(h5_ids, obj=dropped_cells[k])

    Scaled_PCs[k]['specimen_id'] = Scaled_PCs[k]['specimen_id'].astype(str)

#Merge all scaled PC features into one df
data_frames = []
for k in Scaled_PCs.keys():
    data_frames.append(Scaled_PCs[k])

Scaled_PCs = reduce(lambda left, right: pd.merge(left, right, on=['specimen_id'], how='outer'), data_frames)

df = Scaled_PCs.melt(value_vars=Scaled_PCs[[c for c in Scaled_PCs.columns if c != "specimen_id"]])
sns.catplot(x="variable", y="value",kind='box', data=df, palette=sns.color_palette(["skyblue"]),aspect=4.4)
ax = plt.gca()
ax.set(**{'title':'Scaled PC features','xlabel':'','ylabel':''})
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()

print("Scaled PCs size:", Scaled_PCs.shape)
print("...................................................")
print("Loading ipfx features")
ipfx = pd.read_csv(ipfx_file_path)

keep_ipfx_features_bioarxiv = ['ap_1_threshold_v_short_square', 'ap_1_peak_v_short_square',
       'ap_1_upstroke_short_square', 'ap_1_downstroke_short_square',
       'ap_1_upstroke_downstroke_ratio_short_square',
       'ap_1_width_short_square', 'ap_1_fast_trough_v_short_square',
       'short_square_current', 'input_resistance', 'tau', 'v_baseline',
       'sag_nearest_minus_100', 'sag_measured_at', 'rheobase_i',
       'ap_1_threshold_v_0_long_square', 'ap_1_peak_v_0_long_square',
       'ap_1_upstroke_0_long_square', 'ap_1_downstroke_0_long_square',
       'ap_1_upstroke_downstroke_ratio_0_long_square',
       'ap_1_width_0_long_square', 'ap_1_fast_trough_v_0_long_square',
       'avg_rate_0_long_square', 'latency_0_long_square',
       'stimulus_amplitude_0_long_square', "specimen_id"]

ipfx = ipfx[keep_ipfx_features_bioarxiv]

df = ipfx.melt(value_vars=ipfx[[c for c in ipfx.columns if c != "specimen_id"]])
sns.catplot(x="variable", y="value",kind='box', data=df, palette=sns.color_palette(["coral"]),aspect=2.4)
ax = plt.gca()
ax.set(**{'title':'IPFX features','xlabel':'','ylabel':''})
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()

print("...................................................")
print("Zscoring ipfx features")
subset_ipfx = ipfx[[c for c in ipfx.columns if c != "specimen_id"]]
ipfx_norm = (subset_ipfx - subset_ipfx.mean(axis=0)) / subset_ipfx.std(axis=0)

print("Removing extreme ipfx values (within 6 std)")
scaling_thr = ipfx_norm.std(axis=0, skipna=True, numeric_only=True) * 6
ipfx_norm = ipfx_norm.reset_index()
df1 = pd.melt(ipfx_norm, id_vars=['index'], value_vars=[c for c in ipfx_norm if c != "index"])
df2 = pd.DataFrame(scaling_thr).reset_index().rename(columns={"index": 'variable', 0:"thr_std"})
df3 = df1.merge(df2, on="variable")
df3['new_value'] = np.where((df3['value'] < df3['thr_std']) & (df3['value'] > -1 * df3['thr_std'])
                     , df3['value'], np.nan)
ipfx_norm = df3.pivot(index='index', columns="variable", values="new_value")
ipfx_norm['specimen_id'] = ipfx['specimen_id'].astype(str)

df = ipfx_norm.melt(value_vars=ipfx_norm[[c for c in ipfx_norm.columns if c != "specimen_id"]])
sns.catplot(x="variable", y="value",kind='box', data=df, palette=sns.color_palette(["coral"]),aspect=2.4)
ax = plt.gca()
ax.set(**{'title':'IPFX features','xlabel':'','ylabel':''})
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()

print("ipfx features size:", ipfx_norm.shape)

print("...................................................")
print("Merging all E features together")

data_frames = [Scaled_PCs, ipfx_norm]
df_merged = reduce(lambda  left, right: pd.merge(left, right, on=['specimen_id'], how='inner'), data_frames)

annotation = feather.read_dataframe(T_anno_path)
annotation = annotation.rename({"spec_id_label":"specimen_id"}, axis="columns")

df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['specimen_id'], how='inner'), [annotation[["specimen_id", "sample_id"]], df_merged])
df_merged = df_merged.drop(labels=["specimen_id"], axis=1)
df_merged.to_csv(output_path + output_file_prefix + "_" + 'Merged_Ephys_features.csv', index=False)

f = df_merged
df = f.melt(value_vars=f[[c for c in f.columns if c != "sample_id"]])
sns.catplot(x="variable", y="value",kind='box', data=df, palette=sns.color_palette(["skyblue"]),aspect=4.4)
ax = plt.gca()
ax.set(**{'title':'Scaled sPC features','xlabel':'','ylabel':''})
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()

print("Size of Merged E features:", df_merged.shape)
print("Done")