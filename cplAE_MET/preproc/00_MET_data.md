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
# exc_inh_specimen_ids_30Mar22.txt:
# *_cell == 1 => modality data available
# *_cell == 0 => modality data absent

      specimen_id  T_cell  M_cell  E_cell
0     823231829    1.0     0.0     1.0
1     893406540    1.0     0.0     1.0
```


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
(Ref: Olga Gliko) Metadata `m_anno.csv` and arbordensities `hist2d_120x4` are in:
```
/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Patchseq-Exc/dat/m_data/
```
Update `M_cell=1` in `exc_inh_specimen_ids_30Mar22.txt` for available cells

#### 5. MET dataset:
See `data_{i}_proc.ipynb` where {i} is T, E or M