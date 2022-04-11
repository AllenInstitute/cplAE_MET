# Preparing T data:

- The following dataset was used (15885 cells).

      anno = read.feather(/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/patch_seq/star/mouse_patchseq_VISp_20220113_collapsed40_cpm/anno.feather)

- Only cells with ("Core", "I1", "I2", "I3") Tree_call_label were selected (9799).
- Only cells with "mIVSCC-MET" cell_specimen_project_label were selected (7444), if this is not done, other off-pipeline
  cells can be mixed with proper cells.
- Only cells with "structure_label" equal to one of the following values are kept (6510). These regions are withing 
  visual cortex and it is safe to map these cells to the VISp taxonomy.

      "VIS1"      "VIS2/3"    "VIS4"      "VIS5"      "VISa1"     "VISa5"     "VISa6a"    "VISal1"    "VISal2/3"  "VISal4"    "VISal5"    "VISal6a"   "VISam2/3"
      "VISam4"    "VISam5"    "VISam6a"   "VISC5"     "VISl1"     "VISl2/3"   "VISl4"     "VISl5"     "VISl6a"    "VISl6b"    "VISli1"    "VISli2/3"  "VISli4"
      "VISli5"    "VISli6a"   "VISp"      "VISp1"     "VISp2/3"   "VISp4"     "VISp5"     "VISp6a"    "VISp6b"    "VISpl2/3"  "VISpl4"    "VISpl5"    "VISpl6a"   
      "VISpl6b"   "VISpm1"    "VISpm2/3"  "VISpm4"    "VISpm5"    "VISpm6a"   "VISpm6b"   "VISpor1"   "VISpor2/3" "VISpor4"   "VISpor5"   "VISpor6a"  "VISpor6b"
      "VISrl2/3"  "VISrl4"    "VISrl5"    "VISrl6a"  

- Specimen_id of these cells were saved in the following path:

      "/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/git_workspace/cplAE_MET/data/proc/exc_inh_specimen_ids_30Mar22.txt

# Preparing E data:
Next we use the list of obtained specimen ids to check the availability of the E data. In order to see which cells has 
passed the ephys qc. We need to check if the cells where manually or automatically passed or failed. For this we need to 
look into the workflow_state and if something is failed, then flag it. To do so, I wrote a sql query to look into the lims.

- install lims library by cloning the following repo:
  
      http://stash.corp.alleninstitute.org/users/nathang/repos/lims-utils/browse
  
- Navigate to the lims folder (for me the following path):
  
      /allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/git_workspace/lims-utils

- Run the following code to check for the workflow_state of the cell. Make sure you are in the correct conda env with 
  lims package available:

      python -m run_qc --spec_ids_file exc_inh_specimen_ids_30Mar22.txt 
      --spec_ids_filepath  "/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/git_workspace/cplAE_MET/data/proc/"  
      --qc_passed_file  "ephys_qc_exc_inh_specimen_ids_30Mar22.txt"

- The following file which has the workflow state of the cell will be written:
 
      /allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/git_workspace/cplAE_MET/data/proc/ephys_qc_exc_inh_specimen_ids_30Mar22.txt

- Take only the cells that are "manual_passed" or "auto_passed" and save them in the following file (6291):
 
      /allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/git_workspace/cplAE_MET/data/proc/ephys_qc_passed_exc_inh_specimen_ids_30Mar22.txt

- We can now extract the ipfx features and timeseries for the above specimen_ids. To do this please refer to "ipfx_feature_extraction.md"

