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
import cplAE_TE.utils.preproc_helpers as proc_utils
from cplAE_TE.utils.load_helpers import get_paths, load_dataset, load_summary_files


#Read the json file with all input args
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--input", required=True, type=str)
args = parser.parse_args()

with open(args.input) as json_data:
    data = json.load(json_data)

#Input vars
T_data_path = data['input_path'] + data['T_data_file']
T_anno_path = data['input_path'] + data['T_annotation_file']
specimen_path = data['input_path'] + data['specimen_ids_file']
gene_file_path = data['input_path'] + data['gene_file']
beta_threshold = data['beta_threshold']
output_path = data['output_path']
output_file_prefix = data['output_file_prefix']

print("...................................................")
print("Loading input files")
T_dat = feather.read_dataframe(T_data_path)
T_ann = feather.read_dataframe(T_anno_path)

ids = pd.read_csv(specimen_path, header=None)
ids.rename(columns = {0:'specimen_id'}, inplace = True)

T_ann = T_ann.loc[T_ann['spec_id_label'].astype(np.int64).isin(ids['specimen_id'])]
T_ann = T_ann[['spec_id_label',
               'sample_id',
               'Tree_first_cl_id',
               'Tree_first_cl_label',
               'Tree_first_cl_color',
               'Tree_call_label']].reset_index(drop=True)

keep_gene_id = pd.read_csv(gene_file_path)
keep_gene_id = keep_gene_id[keep_gene_id.BetaScore>beta_threshold]['Gene'].to_list()

#Restrict T data based on genes:
keepcols = ['sample_id'] + keep_gene_id
T_dat = T_dat[keepcols]

print("...................................................")
print("Keep data and annotation for given sample_ids")
#Restrict to samples in the annotation dataframe
T_dat = T_dat[T_dat['sample_id'].isin(T_ann['sample_id'])]
T_dat.set_index(keys='sample_id',inplace=True)
T_dat = T_dat.reindex(labels=T_ann['sample_id'])
T_dat.reset_index(drop=False,inplace=True)


print("...................................................")
print("Apply log2 to cpm values")
T_dat[keep_gene_id] = np.log(T_dat[keep_gene_id]+1)


assert (T_ann['sample_id'].sort_index(axis=0) == T_dat['sample_id'].sort_index(axis=0)
       ).all(), 'Order of annotation and data samples is different!'

print("annotation file size:", T_ann.shape)
print("logcpm file size:", T_dat.shape)

print("...................................................")
print("writing output data and annotations")
T_dat.to_csv(output_path + output_file_prefix + "_" + 'T_data.csv',index=False)
T_ann.to_csv(output_path + output_file_prefix + "_" + 'T_annotations.csv',index=False)

print("Done!")