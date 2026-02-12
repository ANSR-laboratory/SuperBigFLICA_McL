#!/bin/sh

import sys
import os
import numpy as np          #1.26.3
import nibabel as nib       #5.2.0
import torch                #2.1.2
import pandas as pd         #2.2.1
import sbf_utils
from sbf_utils import SBF_load, get_model_param, sort_and_filter_by_modality_order, SBF_save_everything



# Load the non-imaging data for SBF

print("Loading the non-imaging data and non-imaging targets for prediction")

# Change the paths to point to your non-imaging data matrices in CSV format, all superfluous columns (like subject ID) removed, no NANs
# Data train and data validation are fed into SuperBigFLICA, data test is fed into get_model_param.py to apply the model from SBF to your new data

def _load_nidp_csv(path):
    # Allow empty cells; downstream filtering will handle missing values.
    data = np.genfromtxt(path, delimiter=",", missing_values="", filling_values=np.nan)
    data = np.atleast_2d(data)
    if data.shape[0] == 1 and data.shape[1] > 1:
        data = data.T
    return data


nIDPs_train = _load_nidp_csv('../forKayla/nIDPs_train_totalMJ.csv')
nIDPs_validation = _load_nidp_csv('../forKayla/nIDPs_validation_totalMJ.csv')
nIDPs_test = _load_nidp_csv('../forKayla/nIDPs_test_totalMJ.csv')


nIDPs_train_npy = nIDPs_train
nIDPs_validation_npy = nIDPs_validation
nIDPs_test_npy = nIDPs_test

# Change the column indexes to pick out the variables for training (e.g., in the train dataset, and the target(s) in the test/validation datasets
indices_to_include = 0
if np.isscalar(indices_to_include):
    indices_to_include = [int(indices_to_include)]
nIDPs_test_targets = nIDPs_test_npy[:, indices_to_include]
nIDPs_train_set = nIDPs_train_npy[:, indices_to_include]
nIDPs_validation_targets = nIDPs_validation_npy[:, indices_to_include]


# Create opts, which is a dictionary of options, filled in with your specific paths, modalities and SBF parameters to execute the SBF data fusion and validation in unseen dataset.

opts = {
    "brain_data_main_folder": "../forKayla",
    "output_dir": "Data/SBF_outputs",
    "modalities_order": ['RF14', 'CT', 'PSA'],
    "modalities_order_filetypes": ['.gz', '.dtseries.nii', '.dtseries.nii'],
    "fs_path": os.getenv("FREESURFER_HOME", "/path/to/freesurfer") + "/bin/freesurfer",
    "path_to_fsavg": os.getenv(
        "FSAVERAGE_PATH",
        os.path.join(os.getenv("FREESURFER_HOME", "/path/to/freesurfer"), "subjects", "fsaverage"),
    ),
    "fsl_path": os.getenv("FSLDIR", "/path/to/fsl") + "/",
    "dropout": 0.2,
    "device": "cpu",
    "auto_weight": [1,1,1,1],
    "lambdas": [.5,.5,.5,.5],
    "nlat": 50,
    "lr": 0.05,
    "random_seed": 42,
    "maxiter": 30,
    "batch_size": 32,
    "init_method": "random",
    "dicl_dim": 250,
    "save_all_epochs": True
}

# Specify folder for data and for SBF outputs
#       brain_data_main_folder: Full path to SBF input data. Organize input data into a folder that contains one sub-folder per modality (FS CT and PSA etc have both lh and rh in one folder),
#                               with a separate folder for each dataset, e.g., VBM_test, VBM_train, VBM_validation each have their own sub-folders, and PSA_train, PSA_test, PSA_valid would have
#                               lh and rh PSA for each dataset within their sub-folders. Also NOTE!!! Freesurfer modalities are assumed to start with lh and rh so don't add anything in front!
#       output_dir: Full path to SBF outputs folder

# Specify what modalities and modality filetypes you're including in the analysis. These will be labels for the output spatial maps files and are needed to organize train, test, and validation inputs the same
#       modalities_order: specify the order of the modalities, e.g., ['PSA', 'CT', 'VBM']. This can be any order you like, as only needed for organizing the inputs into SBF and for writing the output modalities correctly (these names
#                         are the same as the folder names you've given to organize the modalities in the main data folder
#       modalities_order_filetypes: specify the filetypes for each modality in the same order as modalities_order, e.g., for the above example, ['.mgh', '.mgh', '.gz']. The filetypes are the last part of the filename after the last ".".

# Specify paths to software that are needed for SBF
#       fs_path: path to freesurfer
#       path_to_fsavg: path to fsavg that was created by freesurfer when you ran it on the structural images
#       fsl_path: path to fsl

# Other opts include SBF parameters as recommended on Weikang's github
#       dropout: the amount of data that will be dropped out during training steps, which is necessary to avoid overfitting. Recommended is 0.2
#       device: set to cpu or gpu, but note I did not test with gpu
#       auto_weight: whether doing bayes autoweighting among 4 losses (leave it the default [1,1,1,1]).
#       lambdas: weight on different losses (leave it as equal, i.e., [0.25,0.25,0.25,0.25]).
#       lr: learning rate. try 0.001 to 0.05
#       batch_size: default is 512, but note that batch_size needs to be set according to your train dataset size (this is the number of participants/batch
#                   for parameter optimization so it needs to be some increment smaller than your training sample size
#       maxiter: max number of iterations (usually less than 30)


# Outputs of the SBF data fusion and application of the model to the test dataset are:
#       pred_valid: the predicted nIDP in the validation set
#       SBF_best_model: the model that can give the best nIDP prediction in the validation set
#       loss_all_test: the training losses
#       best_corr: the evaluation metric (If there is only one nIDP, the metric is the correlation; if there are multiple nIDPs, the metric is the sum of correlations that are larger than 0.1)
#       SBF_final_model: the model of the last epoch
#       lat_train: the multimodal shared latent variables (subject-by-nlat), use it to correlate/predict other nIDPs.
#       lat_test: the multimodal shared latent variables (subject-by-nlat), use it to correlate/predict other nIDPs.
#       SBF spatial maps: one 4D (number of components) data file for each modality. Each spatial map voxel/vertex values are spatial loadings that have been z-score normalized by regressing the lat_train onto the original data x_train.
#       modality_weights: a nlat-by-modality matrix, it is the contribution of each modality to each latent component.
#       prediction_weights: a nlat-by-#nIDP matrix, the trained weights of predicting each of the nIDPs using the latent components.
#       pred_train, pred_test: the predicted nIDPs by the trained model in training set and test set.



#--------------------Begin Processing (no need to read beyond this unless you want to see the code)----------------------------

# Create the output directory if it does not exist
outdir = opts["output_dir"]
os.makedirs(outdir, exist_ok=True)

# Load in the imaging data and set up the imaging train, test and validation lists of matrices to feed into SBF. That is, SBF expects the train data to be a list of k modality matrices (ibid for test and validation data)
# Note that later on, when applying the model to the validation dataset, get_model_param function expects the train and validation modality datasets to be lists of k torch tensors, so lists of matrices is converted



print("Loading the imaging data")
img_data,img_info,modalities = sbf_utils.SBF_load(opts, nIDPs_train_set, nIDPs_test_targets, nIDPs_validation_targets)

#create headers for SBF output data using the same order as modalities_order and modalities_order_filetypes

# Initialize a list to store headers based on the order of file types in opts
SBFOut_headers = []
SBFOut_affine = []
# SBFOut_shapes = []

# Loop through each file type in opts
for opt_filetype in opts["modalities_order_filetypes"]:
    # Loop through the file types in img_info to find a match
    for index, img_filetype in enumerate(img_info["filetype"]):
        if img_filetype == opt_filetype:
            # If a match is found, append the corresponding header to the list
            # SBFOut_headers.append(img_info["header"][index]) # YC commented
            modality_key = modalities[index] # YC added
            if opt_filetype == '.dtseries.nii':
                # CIFTI outputs use stored axes; no NIfTI header/affine needed here.
                SBFOut_headers.append(None)
                SBFOut_affine.append(None)
            else:
                SBFOut_headers.append(img_info["header"][modality_key]) # YC added
                SBFOut_affine.append(img_info['affine'][modality_key]) # YC added

            break

#add these to opts
opts["SBFOut_headers"] = SBFOut_headers
opts["SBFOut_affine"] = SBFOut_affine
# opts["SBFOut_shapes"] = SBFOut_shapes


# This code reorganizes the img_data into test, train, and validation data. E.g., all modalities for test_data are organized together into a single list of k matrices, ibid for train and validation, and all three lists
# have modalities in the same order.

# Dictionaries to store the indices and data
modality_indices = {}
modality_data = {}

for modality in modalities:
    # Get the index of the current modality
    index = modalities.index(modality)

    # Store the index in the dictionary with a constructed key
    modality_indices[f"{modality}_index"] = index

    # Retrieve the corresponding data using the index and store it
    modality_data[f"{modality}_data"] = img_data[index]

# This code parses the modalities to group together modalities into test, train and validation lists of arrays to pass to SBF


Test_Data_list = [{'modality': key.replace('_data','').replace('_test',''), 'data': data} for key, data in modality_data.items() if 'test' in key]
Train_Data_list = [{'modality': key.replace('_data','').replace('_train',''), 'data': data} for key, data in modality_data.items() if 'train' in key]
Validation_Data_list = [{'modality': key.replace('_data','').replace('_validation',''), 'data': data} for key, data in modality_data.items() if 'validation' in key]

# Test, Train, and Validation Data lists will likely not have modalities in the same order, so reorganize them to have the same order as in opts.modalities_order

# First, clean suffixes off the modality names in Test_, Train_, and Validation_Data_list(s). Having these in the data folder names helps organize all, e.g., test dataset modalities
# together, but then to make sure the final lists all have the same modality order, we need to remove these suffixes and compare the modality orders



# Sort each data list
order = opts["modalities_order"]
Test_Data_list = sort_and_filter_by_modality_order(Test_Data_list, order)
Train_Data_list = sort_and_filter_by_modality_order(Train_Data_list, order)
Validation_Data_list = sort_and_filter_by_modality_order(Validation_Data_list, order)

# Extracting the data matrices into a new list to feed into SBF; now train, test and validation datasets will have exactly the same order of the modalities
Data_train = [item['data'] for item in Train_Data_list]
Data_test = [item['data'] for item in Test_Data_list]
Data_validation = [item['data'] for item in Validation_Data_list]





print("Starting SuperBigFLICA")

pred_best, best_model, loss_all_test, best_corr, final_model, Data_train, Data_validation, Data_test, dictionaries = sbf_utils.SupervisedFLICA(x_train = Data_train, y_train = nIDPs_train_set, x_test = Data_validation, y_test = nIDPs_validation_targets, 
Data_test = Data_test,
output_dir = opts["output_dir"],
dropout = opts["dropout"], 
device = opts["device"], 
auto_weight = opts["auto_weight"], 
lambdas = opts["lambdas"],
nlat = opts["nlat"], 
lr = opts["lr"], 
random_seed = opts["random_seed"], 
maxiter = opts["maxiter"], 
batch_size = opts["batch_size"], 
init_method = opts["init_method"],
dicl_dim = opts["dicl_dim"],
save_all_epochs = opts.get("save_all_epochs", False),
img_info = img_info,
opts = opts)

print("Saving SBF Model Outputs")
outdir = opts["output_dir"]
fname = "pred_valid.csv"
fname_fullpath = os.path.join(outdir, fname)
np.savetxt(fname_fullpath, pred_best, delimiter=',')

fname = "SBF_best_model.pth"
fname_fullpath = os.path.join(outdir, fname)
torch.save(best_model.state_dict(), fname_fullpath)

fname = "loss_all_test.csv"
fname_fullpath = os.path.join(outdir, fname)
np.savetxt(fname_fullpath, loss_all_test, delimiter=',')


fname = "best_corr.txt"
fname_fullpath = os.path.join(outdir, fname)
# Write to the file using the full path
with open(fname_fullpath, 'w') as file:
    file.write(str(best_corr))

fname = "SBF_final_model.pth"
fname_fullpath = os.path.join(outdir, fname)
torch.save(final_model.state_dict(), fname_fullpath)

# from utils3 import get_model_param #, grid_search_elasticnet

print("Starting application of the model to the test dataset")

# First convert train and test to lists of tensors (data fusion requires lists of arrays, but get_model_param requires lists of tensors)
Data_train = [torch.tensor(matrix) for matrix in Data_train]
Data_test = [torch.tensor(matrix) for matrix in Data_test]

lat_train, lat_test, spatial_loadings, modality_weights, prediction_weights, pred_train, pred_test, best_performance = get_model_param( x_train = Data_train, x_test = Data_test, y_train = nIDPs_train_set, y_test = nIDPs_test_targets,best_model = best_model)



print("Saving SuperBigFLICA outputs from get_model_param application to new test data")




file_data_pairs = [
    ("lat_train.csv", lat_train),
    ("lat_test.csv", lat_test),
    ("modality_weights.csv", modality_weights),
    ("prediction_weights.csv", prediction_weights),
    ("pred_train.csv", pred_train),
    ("pred_test.csv", pred_test),
    ("best_performance.csv", np.array(best_performance))
]


# Save each file using a loop
for fname, data in file_data_pairs:
    np.savetxt(os.path.join(outdir, fname), data, delimiter=',')



spatial_maps = SBF_save_everything(outdir, spatial_loadings, img_info, opts, dictionaries)

print("Done!! All outputs of data fusion and application to test data can be viewed in the SBF output directory!")
