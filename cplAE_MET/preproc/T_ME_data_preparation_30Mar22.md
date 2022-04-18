# Preparing T, E and M data for feature extraction.

## Directory for data:

The up-to-date data are saved in the following path:
 
    /allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Patchseq-Exc/

## 1- Getting specimen_ids:

All the exc and inh patchseq cells that passed QC are saved in this "exc_inh_specimen_ids_30Mar22.txt" in the folder 
mentioned above. These set of cells contain all the exc and inh in the published inh paper and going-to-be-published exc 
paper (only full morpho cells are not included in this list yet). There might be 50 cells that were initially
in the locked dataset that are not in this list and the reason is that in the new mapping available, those cells were discarded
with one of the filters that are explained below. How these specimen ids were obtained?

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

- At the end, cells that have M data available but they were poorQ cells, were added back to the dataset because we are going 
  to treat these cells as M_only cells with no E and T. In total 60 cells were added back to the above cells (6570). For E cells 
  we dont do this because as of now, E cells that have PoorQ T type, are not reliable.
  
- Specimen_id of these cells were saved in the following file:

      "//allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Patchseq-Exc/dat/exc_inh_specimen_ids_30Mar22.txt

NOTE: to distinguish cells that have T-only or E-only or M-only data available. In "exc_inh_specimen_ids_30Mar22.txt" file
we added three columns called "T_cell", "E_cell" and "M_cell". Each of these columns can be 1.0 or 0.0. if it is 1.0 it
indicates that modality of data is available for that specimen id.

## 2- Get specimen ids for the ephys-QC-passed cells:

Now from the above list of cells, we should check their ephys quality and get the specimen ids for those that passed QC.
In order to see which cells has passed the ephys QC. We need to check if the cells were manually or automatically passed 
or failed. For this we need to look into the workflow_state table in lims and if something is failed, then flag it. 
To do so, I wrote a sql query to look into the lims data. To run that, we need to first install lims. You can do this 
using the following:

- install lims library by cloning the following repo:
  
      http://stash.corp.alleninstitute.org/users/nathang/repos/lims-utils/browse
  
- Navigate to the lims folder (for me the following path):
  
      /allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/git_workspace/lims-utils

- Run the following code to check for the workflow_state of the cell. Make sure you are in the correct conda env with 
  lims package available:

      python -m run_qc --spec_ids_file exc_inh_specimen_ids_30Mar22.txt 
      --spec_ids_filepath  "/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Patchseq-Exc/dat/"  
      --qc_passed_file  "ephys_qc_exc_inh_specimen_ids_30Mar22.txt"

- The following file which has the workflow state of the cell will be written:
 
      /allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Patchseq-Exc/dat/ephys_qc_exc_inh_specimen_ids_30Mar22.txt

- Take only the cells that are "manual_passed" or "auto_passed" and save them in the following file (6291 cells):
 
      /allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Patchseq-Exc/dat/ephys_qc_passed_exc_inh_specimen_ids_30Mar22.txt

## 3- Extract ipfx and ephys timeseries:

We can now extract the ipfx features and timeseries for the above specimen_ids. To do this please refer to 
"ipfx_feature_extraction.md". From 6291 cells that passed ephys-QC, and went through the ipfx feature extraction pipeline,
5854 cells had features available. So now for these cells, we update the "E_cell" columns of "exc_inh_specimen_ids_30Mar22.txt" 
file. If feature is available, then that column is 1.0 otherwise 0.0.

## 4- M arbor densities:
This data is provided by Olga. In the following folder there is a file which has the meta_data and it is called
m_anno.csv. 

     /allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Patchseq-Exc/dat/m_data

and there is a folder which has all the arbor densities calculated. For all the cells that have M data available, you need 
to put M_cell equal to 1.0 in the "exc_inh_specimen_ids_30Mar22.txt" file.

Now all these T, M and E data can be used to generate the mat file which is the input to the T_ME coupled autoencoder.
To do this, please refer to the data_{i}_proc.ipynb where {i} is T, E or M 






