#!/usr/bin/env python

import os
import bob

N_MAX_FILES_PER_JOB = 100
N_MAX_PROBES_PER_JOB = 1000

# 0/ The database to use
db = bob.db.banca.Database()
protocol = 'P'
base_output_USER_dir = "/idiap/user/lelshafey/banca/pca"
base_output_TEMP_dir = "/idiap/temp/lelshafey/banca/pca"

# 1/ Face normalization
img_input_dir = "/idiap/group/vision/visidiap/databases/banca/english/images_gray"
img_input_ext = ".pgm"
pos_input_dir = "/idiap/group/vision/visidiap/databases/groundtruth/banca/english/eyecenter"
pos_input_ext = ".pos"
features_dir = os.path.join(base_output_TEMP_dir, "preprocessed")
features_ext = ".hdf5"
first_annot = 0
all_files_options = {}

# Cropping
CROP_EYES_D = 33
CROP_H = 80
CROP_W = 64
CROP_OH = 16
CROP_OW = 32

# Tan Triggs
GAMMA = 0.2
SIGMA0 = 1.
SIGMA1 = 2.
SIZE = 5
THRESHOLD = 10.
ALPHA = 0.1

# 2/ Train PCA matrix
world_options = { }
pca_model_filename = os.path.join(base_output_TEMP_dir, "pca_model.hdf5")
pca_n_outputs = 300

# 3/ Project features using train PCA matrix
featuresProjected_dir = os.path.join(base_output_TEMP_dir, "featuresProjected")
featuresProjected_ext = ".hdf5"

# 4/ PCA models  
models_dir = os.path.join(base_output_TEMP_dir, protocol, "models")
tnorm_models_dir = os.path.join(base_output_TEMP_dir, protocol, "tmodels")

# 5/ Compute scores
scores_nonorm_dir = os.path.join(base_output_USER_dir, protocol, "scores", "nonorm")
scores_ztnorm_dir = os.path.join(base_output_USER_dir, protocol, "scores", "ztnorm")

zt_norm = True

zt_norm_A_dir = os.path.join(base_output_TEMP_dir, protocol, "zt_norm_A")
zt_norm_B_dir = os.path.join(base_output_TEMP_dir, protocol, "zt_norm_B")
zt_norm_C_dir = os.path.join(base_output_TEMP_dir, protocol, "zt_norm_C")
zt_norm_D_dir = os.path.join(base_output_TEMP_dir, protocol, "zt_norm_D")
zt_norm_D_sameValue_dir = os.path.join(base_output_TEMP_dir, protocol, "zt_norm_D_sameValue")
