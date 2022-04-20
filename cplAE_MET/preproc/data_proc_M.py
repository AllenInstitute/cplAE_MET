#########################################################
################ Preprocessing M data ###################
#########################################################
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from cplAE_MET.models.augmentations import undo_radial_correction, do_radial_correction
from cplAE_MET.utils.load_config import load_config


parser = argparse.ArgumentParser()
parser.add_argument('--config_file',    default='config_preproc.toml', type=str,   help='config file with data paths')



def set_paths(config_file=None):
    paths = load_config(config_file=config_file, verbose=False)
    paths['input'] = f'{str(paths["package_dir"] / "data/proc/")}'
    paths['output'] = f'{paths["input"]}/{str(paths["m_output_file"])}'

    paths['specimen_ids'] = f'{paths["input"]}/{str(paths["specimen_ids_file"])}'
    paths['m_data_folder'] = f'{paths["input"]}/{str(paths["m_data_folder"])}'
    paths['m_anno'] = f'{paths["m_data_folder"]}/{str(paths["m_anno"])}'
    paths['hist2d_120x4'] = f'{paths["m_data_folder"]}/{str(paths["hist2d_120x4_folder"])}'

    paths['t_anno'] = f'{paths["input"]}/{"anno.feather"}'

    return paths


def get_file_apendix(exc_or_inh):
    appendix = []
    if exc_or_inh == "inh":
        appendix = ["axon", "dendrite"]
    if exc_or_inh == "exc":
        appendix = ["apical", "basal"]
    return appendix


def get_cell_ids_of_abnormal_images(specimen_ids, image_path, m_anno,  min_nonzero_pixels=5):
    '''
    Get all the specimen_ids that have few nonzero pixels

    Args:
        anno: annotation file, which has column called specimen_id
        image_path: the path to the images
        exc_or_inh: inh or exc cells are being processed
        min_nonzero_pixels: the cell is abnormal if it has less than this number of nonzero pixels

    Returns:
        list of specimen ids of the abnormal cell images
    '''

    ab_spec_id = []
    for i, spec_id in tqdm(enumerate(specimen_ids)):
        if spec_id in m_anno['specimen_id'].to_list():
            exc_or_inh = m_anno[m_anno['specimen_id'] == spec_id]['class']
            app = get_file_apendix(exc_or_inh.values[0])
            if os.path.isfile(image_path + f'/hist2d_120x4_{app[0]}_{spec_id}.csv'):

                im0 = pd.read_csv(image_path + f'/hist2d_120x4_{app[0]}_{spec_id}.csv', header=None).values
                im1 = pd.read_csv(image_path + f'/hist2d_120x4_{app[1]}_{spec_id}.csv', header=None).values

                if np.count_nonzero(im0) < min_nonzero_pixels or np.count_nonzero(im1) < min_nonzero_pixels:
                    ab_spec_id.append(spec_id)
    return ab_spec_id

def main(config_file='config_preproc.toml'):


    dir_pth = set_paths(config_file=config_file)
    m_anno_path = dir_pth['m_anno']
    # t_anno_path = dir_pth['t_anno']
    hist2d_120x4_path = dir_pth["hist2d_120x4"]

    ################## Reading m_anno and finding cells with few nonzero pixels
    ids = pd.read_csv(dir_pth['specimen_ids'])
    specimen_ids = ids['specimen_id'].tolist()
    m_anno = pd.read_csv(m_anno_path) #This is used for soma depth and class type
    ab_spec_id = get_cell_ids_of_abnormal_images(specimen_ids, hist2d_120x4_path, m_anno,  min_nonzero_pixels=5)
    print(len(ab_spec_id), "cells will be dropped because of the few non zero pixels")
    drop_spec_id = ab_spec_id

    ################## Read the t data and find the cells with poor Q
    # t_anno = feather.read_dataframe(t_anno_path)
    # t_anno = t_anno.rename(columns={"spec_id_label": "specimen_id"})
    #t_anno['specimen_id'] = t_anno['specimen_id'].astype(int)

    #t_poorQ_ids = t_anno[(t_anno["Tree_call_label"] == "PoorQ")]["specimen_id"].to_list()
    #t_poorQ_ids = [i for i in t_poorQ_ids if i in specimen_ids]
    #drop_spec_id = ab_spec_id + t_poorQ_ids

    ################## Find small clusters to remove from m data
    #specimen_ids = [i for i in specimen_ids if i not in drop_spec_id]
    #sub_t_anno = t_anno[t_anno["specimen_id"].isin(specimen_ids)]
    #rm_small_clusters = [k for k, v in Counter(sub_t_anno['Tree_first_cl_label']).items() if v <= 5]
    #small_cluster_cells = sub_t_anno[sub_t_anno["Tree_first_cl_label"].isin(rm_small_clusters)]['specimen_id'].to_list()
    #drop_spec_id = np.unique(drop_spec_id + small_cluster_cells)
    #print(len(drop_spec_id),"in total this amount of cells will be dropped from m data")

    ################## Drop those ids from the anno file
    # specimen_ids = [i for i in specimen_ids if i not in drop_spec_id]
    # df_spec = pd.DataFrame(specimen_ids, columns=["specimen_id"])

    ################## Merge with t_anno and make sure the order is the same as in the specimen id file
    # t_anno = t_anno.merge(df_spec, on="specimen_id", how='right')
    # t_anno = t_anno[["specimen_id", "Tree_first_cl_label", "Tree_first_cl_color", "Tree_first_cl_id"]]

    print("...................................................")
    print("Generating image for all the locked dataset, for those that we dont have M, we put nan")
    hist_shape = (1, 120, 4, 1)
    im_shape = (1, 120, 4, 4)
    im = np.zeros((ids.shape[0], 120, 4, 4), dtype=float)
    soma_depth = np.zeros((ids.shape[0],))
    c = 0
    for i, spec_id in tqdm(enumerate(ids['specimen_id'])):
        if spec_id in drop_spec_id:
            im[i, ...] = np.full(im_shape, np.nan)
            soma_depth[i] = np.nan
        else:
            if spec_id in m_anno['specimen_id'].to_list():
                exc_or_inh = m_anno[m_anno['specimen_id'] == spec_id]['class'].values[0]
                app = get_file_apendix(exc_or_inh)
                if os.path.isfile(hist2d_120x4_path + f'/hist2d_120x4_{app[0]}_{spec_id}.csv'):
                    c += 1
                    im0 = pd.read_csv(hist2d_120x4_path + f'/hist2d_120x4_{app[0]}_{spec_id}.csv', header=None).values
                    im1 = pd.read_csv(hist2d_120x4_path + f'/hist2d_120x4_{app[1]}_{spec_id}.csv', header=None).values

                    #convert arbor density to arbor mass
                    mass0 = undo_radial_correction(im0)
                    mass1 = undo_radial_correction(im1)

                    # Normalize so that the mass sum is 350
                    mass0 = mass0 * 350 / np.sum(mass0)
                    mass1 = mass1 * 350 / np.sum(mass1)

                    #compute the arbor density from the arbor mass again
                    im0 = do_radial_correction(mass0)
                    im1 = do_radial_correction(mass1)

                    if exc_or_inh == "inh":
                        im[i, :, :, 0:2] = (np.concatenate([im0.reshape(hist_shape), im1.reshape(hist_shape)], axis=3))
                    else:
                        im[i, :, :, 2:] = (np.concatenate([im0.reshape(hist_shape), im1.reshape(hist_shape)], axis=3))

                    soma_depth[i] = np.squeeze(m_anno.loc[m_anno['specimen_id'] == spec_id]['soma_depth'].values)
                else:
                    im[i, ...] = np.full(im_shape, np.nan)
                    soma_depth[i] = np.nan
            else:
                im[i, ...] = np.full(im_shape, np.nan)
                soma_depth[i] = np.nan

    print("so far in total", c, "cells have m data available")

    sio.savemat(dir_pth['output'], {'hist_ax_de_api_bas': im,
                                    'soma_depth': soma_depth,
                                    'specimen_id': ids['specimen_id'].to_list()}, do_compression=True)

    return

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
