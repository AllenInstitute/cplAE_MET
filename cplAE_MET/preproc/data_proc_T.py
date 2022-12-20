#########################################################
################ Preprocessing T data ###################
#########################################################
import feather
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from cplAE_MET.utils.load_config import load_config


parser = argparse.ArgumentParser()
parser.add_argument('--config_file',    default='config_preproc.toml', type=str,   help='config file with data paths')
parser.add_argument('--beta_threshold', default=0.4,                   type=float, help='beta threshold for removing genes')



def set_paths(config_file=None):
    paths = load_config(config_file=config_file, verbose=False)

    paths['input'] = f'{str(paths["data_dir"])}'
    paths['arbor_density_file'] = f'{paths["input"]}/{str(paths["arbor_density_file"])}'
    paths['arbor_density_PC_file'] = f'{paths["input"]}/{str(paths["arbor_density_PC_file"])}'

    paths['specimen_ids'] = f'{paths["input"]}/{str(paths["specimen_ids_file"])}'
    paths['gene_file'] = f'{paths["input"]}/{str(paths["gene_file"])}'

    paths['t_anno'] = f'{paths["input"]}/{"anno.feather"}'
    paths['t_data'] = f'{paths["input"]}/{"data.feather"}'
    paths['t_data_output'] = f'{paths["input"]}/{str(paths["t_data_output_file"])}'
    paths['t_anno_output'] = f'{paths["input"]}/{str(paths["t_anno_output_file"])}'
    paths['gene_id_output'] = f'{paths["input"]}/{str(paths["gene_id_output_file"])}'

    return paths



def main(config_file='config_preproc.toml', beta_threshold=0.4):


    dir_pth = set_paths(config_file=config_file)
    gene_file_path = dir_pth['gene_file']
    beta_threshold = beta_threshold


    print("...................................................")
    print("Loading input files")
    T_dat = feather.read_dataframe(dir_pth['t_data'])
    T_ann = feather.read_dataframe(dir_pth['t_anno'])

    ids = pd.read_csv(dir_pth['specimen_ids'])
    ids['specimen_id'] = ids['specimen_id'].astype(str)
    T_ann['spec_id_label'] = T_ann['spec_id_label'].astype(str)
    t_cells = [i for i in ids['specimen_id'].to_list() if i in T_ann['spec_id_label'].to_list()]
    specimen_ids = ids['specimen_id'].tolist()
    not_t_cells = [i for i in specimen_ids if i not in t_cells]
    print("...................................................")
    print("There are", len(specimen_ids), "sample_ids in the locked dataset")

    T_ann['spec_id_label'] = T_ann['spec_id_label'].astype(np.int64)
    T_ann = T_ann.rename(columns={"spec_id_label": "specimen_id"})
    df_spec_id = pd.DataFrame(specimen_ids, columns=["specimen_id"])
    df_spec_id['specimen_id'] = df_spec_id['specimen_id'].astype(str)
    T_ann['specimen_id'] = T_ann['specimen_id'].astype(str)
    T_ann = T_ann.merge(df_spec_id, on="specimen_id", how='right')
    T_ann = T_ann[['specimen_id',
                   'sample_id',
                   'Tree_first_cl_id',
                   'Tree_first_cl_label',
                   'Tree_first_cl_color',
                   'Tree_call_label']].reset_index(drop=True)


    counts = Counter(T_ann['Tree_call_label'])
    print("There are", counts['Core'] + counts['I1'], "highly consistent cells")
    print("There are", counts['I2'] + counts['I3'], "Moderately consistent cells")
    print("There are", counts['PoorQ'], "Inconsistent cells")
    poorQ = T_ann[T_ann['Tree_call_label']=="PoorQ"]['specimen_id'].to_list()
    t_cells = [i for i in t_cells if i not in poorQ]
    not_t_cells = not_t_cells + poorQ
    
    keep_gene_id = pd.read_csv(gene_file_path)
    keep_gene_id = keep_gene_id[keep_gene_id.BetaScore>beta_threshold]['Gene'].to_list()

    #Restrict T data based on genes:
    keepcols = ['sample_id'] + keep_gene_id
    T_dat = T_dat[keepcols]

    print("...................................................")
    print("Keep data and annotation for given sample_ids")
    #Restrict to samples in the annotation dataframe
    T_dat = T_dat.merge(T_ann[['sample_id', 'specimen_id']], on="sample_id", how='right')
    T_dat = T_dat.drop(labels=["sample_id"], axis=1)
    T_dat = T_dat.set_index("specimen_id")
    T_dat = T_dat.reset_index()

    print("...................................................")
    print("set the cpm values for non tcells to nan")
    T_dat = T_dat.set_index("specimen_id")
    T_ann = T_ann.set_index("specimen_id")
    T_dat.loc[not_t_cells] = np.nan
    T_ann.loc[not_t_cells] = np.nan

    print("...................................................")
    print("Apply log2 to cpm values for t_cells only")
    T_dat[keep_gene_id] = np.log(T_dat[keep_gene_id]+1)


    assert (T_dat.index.to_list() == specimen_ids), \
        'Order of data samples and id list is different!'

    assert (T_ann.index.to_list() == specimen_ids), \
        'Order of annotation id list is different!'

    T_dat = T_dat.reset_index()
    T_ann = T_ann.reset_index()
    print("annotation file size:", T_ann.shape)
    print("logcpm file size:", T_dat.shape)

    print("...................................................")
    print("writing output data and annotations")
    T_dat.to_csv(dir_pth['t_data_output'], index=False)
    T_ann.to_csv(dir_pth['t_anno_output'], index=False)

    keep_gene_id = pd.DataFrame(keep_gene_id, columns=["gene_id"])
    keep_gene_id.to_csv(dir_pth['gene_id_output'], index=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))



