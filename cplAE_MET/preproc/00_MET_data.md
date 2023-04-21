This document describes steps to assemble specimen ids for which M, E, T data is available. Output of these steps is in: `/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Patchseq-Exc/`

#### 1. Get specimen_ids:

Specimen ids for QC passing exc and inh cells (both publushed and unpublished data): `exc_inh_specimen_ids_30Mar22.txt`

 - Get metadata from `/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/patch_seq/star/mouse_patchseq_VISp_20220113_collapsed40_cpm/anno.feather`
- Filter `tree_call_label in {"Core", "I1", "I2", "I3"}`
- Filter `cell_specimen_project_label=="mIVSCC-MET"`
- Filter `structure_label in region_set` where:
```python
region_set = {"VIS1", "VIS2/3", "VIS4", "VIS5",
              "VISa1", "VISa5", "VISa6a", "VISal1",
              "VISal2/3", "VISal4", "VISal5", "VISal6a",
              "VISam2/3", "VISam4", "VISam5", "VISam6a",
              "VISC5", "VISl1", "VISl2/3", "VISl4", "VISl5",
              "VISl6a", "VISl6b", "VISli1", "VISli2/3",
              "VISli4", "VISli5", "VISli6a", "VISp", "VISp1",
              "VISp2/3", "VISp4", "VISp5", "VISp6a", "VISp6b",
              "VISpl2/3", "VISpl4", "VISpl5", "VISpl6a",
              "VISpl6b", "VISpm1", "VISpm2/3", "VISpm4",
              "VISpm5", "VISpm6a", "VISpm6b", "VISpor1",
              "VISpor2/3" "VISpor4", "VISpor5", "VISpor6a",
              "VISpor6b", "VISrl2/3", "VISrl4", "VISrl5", "VISrl6a"}
```
- 6510 specimen ids remain at the end of all above filtering steps
- ~60 cells with M that do not pass T QC were appended separately (T and E for these cells are treated as absent)
- We do not follow a similar procedure for E cells yet

```
# exc_inh_ME_fMOST_EM_specimen_ids_13Mar23.txt:
# *_cell == True => modality data available
# *_cell == False => modality data absent

      specimen_id platform  T_cell  E_cell  M_cell
0     823231829   patchseq  True    False   True
1     893406540   patchseq  True    True    True
```
So far specimen ids from patchseq data are extracted. 
- For the ME cells. We have a file which shows the spec_ids and metadata for the ME cells and there are 1946 cells in that file:
`/home/fahimehb/Remote-AI-root/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Patchseq-Exc/dat/ME_dataset.csv`

None of these cells have T data available. Some have M, some have E and some other have both M and E. Some of these cells were 
excluded during next step (QC for E). The list of cells that were excluded are: ['325941643', '326774520', '333604946', '468193142', '476135066', '476216637', '485161419', '497611660', '501850666', '515464483', '526643573', '526668864', '569965244', '590558808', '592952680']. All the specimen ids of the ME cells except these cells were included in the "exc_inh_ME_fMOST_EM_specimen_ids_13Mar23.txt" file.

- For the fmost cells, as they only have M data available. added those that Olga could calculate arbor density for them. 

#### 2. QC for E cells:

This requires an sql query on the `workflow_state` table in lims. Steps are below:
1. clone and install lims library into a conda environment:
```
http://stash.corp.alleninstitute.org/users/nathang/repos/lims-utils/browse
```
  
2. Navigate to local lims library and activate conda environment:
```
/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/git_workspace/lims-utils
```

3. Check workflow_state of the cell
```bash
    python -m run_qc \
        --spec_ids_file "exc_inh_specimen_ids_30Mar22.txt" \
        --spec_ids_filepath "/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Patchseq-Exc/dat/" \
        --qc_passed_file "ephys_qc_exc_inh_specimen_ids_30Mar22.txt"
```

Specimen ids for "manual_passed" or "auto_passed" cells are saved to:
```
/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Patchseq-Exc/dat/ephys_qc_passed_exc_inh_specimen_ids_30Mar22.txt
```

#### 3. ipfx features and timeseries for QC passing E cells:

 - See `01_E_features.md` to get ipfx and timeseries for `E_cell==1` specimen ids.
 - Features calculation fails for some cells (various reasons)
 - Update `E_cell` column in `exc_inh_specimen_ids_30Mar22.txt` ( = 0 for failed cells)


#### 4. M data:
Patchseq aspiny cells arbor densities (543 manual and 280 auto-traced, 823 unique):
`/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Olga/morphology/arbor_density_ae/aspiny082421`
Patchseq spiny cells arbor densities (817 auto and 436 auto-traced and some are repeated in the two files, in total 831 unique):
`/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Olga/morphology/arbor_density_ae/spiny`
In total there are 1654 unique cells from patchseq

Then we have 282 full morphology spiny cells in the following directory:
`/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Olga/morphology/arbor_density_ae/full_morphology`

Then we have 203 inh and 244 exc ME cells(447 unique cells):
`/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Olga/morphology/arbor_density_ae/ME_cells`

Then we have 362 exc and 141 inh EM cells(503 unique cells):
`/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Olga/morphology/arbor_density_ae/em_data`

So in total 2886 cells have arbor densities calculated.

(Ref: Olga Gliko) Metadata `m_anno_13Mar23.csv` and arbordensities `hist2d_120x4` are in:
```
/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Patchseq-Exc/dat/m_data/
```

#### 5. MET dataset:
See `data_{i}_proc.ipynb` where {i} is T, E or M