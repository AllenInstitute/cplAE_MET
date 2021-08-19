#########################################################
############### Preprocessing M data ####################
#########################################################

import json
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pylab as plt
import cplAE_MET.utils.analysis_helpers as analysis
from sklearn.preprocessing import normalize

#Read the json file with all input args
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--input", required=True, type=str)
args = parser.parse_args()

with open(args.input) as json_data:
    data = json.load(json_data)

M_data_path = data['input_path'] + "/" + data["M_data_file"]
M_data_output_filename = data["M_output_data_file"]
output_path = data['output_path']

print("...................................................")
print("Loading M data")
M_dat = sio.loadmat(M_data_path)
hist_ax_de = np.moveaxis(M_dat["hist_ax_de"], -1,1)

print("...................................................")
print("removing nans from the data")

hist_ax_de, mask_ax_de = analysis.remove_nan_observations(hist_ax_de)

print("size of the data after removing nans:")
print(hist_ax_de.shape)

print("Normalizing ax and de data")

#split the data for ax and de
ax, de = np.split(hist_ax_de, 2, axis=1)

#normalize ax and de separately
def normalize_de_ax(x):
    shape = x.shape
    x = x.reshape(x.shape[0], -1)
    x_norm = normalize(x, axis=1, norm='l1')
    x_norm = x_norm*10
    x_norm = x_norm.reshape(x_norm.shape[0], *shape[1:])
    return x_norm

ax = normalize_de_ax(ax)
de = normalize_de_ax(de)

print("Putting back nans")
#put back nans and bring the ax and de to their original shape
ax = analysis.convert_to_original_shape(ax, mask_ax_de)
de = analysis.convert_to_original_shape(de, mask_ax_de)

print("Concating ax and de data")
#concat the ax and de to save to the final mat
hist_ax_de = np.concatenate((ax,de), axis=1)

print("...................................................")
print("Size of the output data")
print(hist_ax_de.shape)

output_mat = {}
output_mat["hist_ax_de"] = hist_ax_de
output_mat["soma_depth"] = M_dat['soma_depth']


print("...................................................")
print("Writing the output file")
sio.savemat(output_path + "/" + M_data_output_filename , output_mat)
print("Done!")

