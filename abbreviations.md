## Explanation of Model (MATLAB) Keys

### Input of autoencoder models:

- **XT** (*T_dat*): log2(CPM + 1) for genes above specified beta threshold amongst cells from the master list of all methods that are also in the specific patch_seq `anno.feather` table (the "locked dataset"), after having low-quality cells removed. Removed/missing cells are assigned NaN.

- **XM** (*M_dat*): Normalized arbor mass for cells in the master list that have a corresponding morphological annotation. The array is dvided based on whether the cell is excitory or inhibitory, with excitory cells occupying the first two slots of the last axis and inhibitory cells occupying the last two slots (with zeros filling unused slots).

- **XE** (*E_dat*): Combination of whitened PCA factors from the electrophysiological timeseries experiments, and ipfx features extracted from the same timeseries data.

- **is_t_1d**: Boolean array indicating which samples have at least one transcriptomic (XT) feature that is not NaN.

- **is_e_1d**: Boolean array indicating which samples have at least one electrophysiological (XE) feature that is not NaN.

- **is_m_1d**: Boolean array indicating which samples have at least one morphological (XM) feature that is not NaN.

### Output of autoencoder models

- **XrT**: Reconstruction of transcriptomic data (XT) from the corresponding autoencoder.

- **XrE**: Reconstruction of electrophysiological data (XE) from the corresponding unpaired autoencoder.

- **XrE_me_paired**: Reconstruction of electrophysiological data (XE) from the corresponding paired autoencoder.

- **XrM**: Reconstruction of morphological data (XM) from the corresponding unpaired autoencoder.

- **XrM_me_paired**: Reconstruction of morphological data (XM) from the corresponding paired autoencoder.

- **zt**: Latent representation of transcriptomic data (XT) from the corresponding autoencoder.

- **zm**: Latent representation of morphological data (XM) from the corresponding unpaired autoencoder.

- **ze**: Latent representation of electrophysiological data (XE) from the corresponding paired autoencoder.

- **zme_paired**: Joint latent representation of morphological (XM) and electrophysiological (XE) data from the paired autoencoder.

### Metadata for cell samples

- **cluster_id** (*cluster_id*): An ID number based on the transcriptomic classification type of the cell.

- **gene_ids** (*gene_ids*): Names of the genes that were measured in the transcriptomic (XT) data.

- **e_features** (*E_features*): Names of the electrophysiological (XE) features (column names of the dataframe)

- **specimen_id** (*specimen_id*): IDs for the individual cells from which data was gathered.

- **cluster_label** (*cluster_label*): The type name at the leaf of the classification hierarchy tree.

- **cluster_color** (*cluster_color*): Hex code for the color that should be associated with the cell type in a plot.

- **merged_cluster_label_atXX** (*merged_cluster_label_atXX*): Cell class label when the classificaton tree is agglomerated into XX classes.

- **platform** (*platform*): The method by which the cell was probed (e.g. Patch-seq, EM, ME).

- **class** (*class*): Whether the cell is excitory or inhibitory.

- **class_id** (*class_id*): Binary encoding of class, with 0 for excitory and 1 for inhibitory.

- **group** (*group*): TBA, binary label of some kind.

- **subgroup** (*subgroup*): TBA, binary label of some kind.

### Other

- **hist_ax_de_api_bas** (*hist_ax_de_api_bas*): Duplicate of XM.

- **train_ind**: Indices for the cells used to train the model.

- **val_ind**: Indices for the cells used to validate the model.

## Explanation of metrics from TensorBoard (loss dict) 

- **MSE_XT** (*rec_t*): Reconstruction mean-square error for the transcriptomic data (XT).

- **MSE_XM** (*rec_m*): Reconstruction mean-square error for the morphological data (XM).

- **MSE_XE** (*rec_e*): Reconstruction mean-square error for the electrophysiological data (XT).

- **MSE_M_XME** (*rec_m_me*): Reconstruction mean-square error for the morphological data (XM) using the paired autoencoder.

- **MSE_E_XME** (*rec_e_me*): Reconstruction mean-square error for the electrophysiological data (XE) using the paired autoencoder.

- **cpl_1->2** (*cpl_1->2*): Mean-square error of modality 1 and modality 2 latent representations, scaled by the smallest PCA variance, with gradients propogating only to the network generating the representation for modality 1 (so 1->2 is not operationally equivalent to 2->1). Modalities symbols are: t = transcriptomic, e = electrophysiological, m = morphological, me = joint morphological-electrophysiological