#!/usr/bin/env python

import os
import bob

N_MAX_FILES_PER_JOB = 100
N_MAX_PROBES_PER_JOB = 1000

# 0/ The database to use
db = bob.db.banca.Database()
protocol = 'P'
base_output_TEMP_dir = "/idiap/temp/lelshafey/banca/tantriggs"

# 1/ Face normalization
img_input_dir = "/idiap/group/vision/visidiap/databases/banca/english/images_gray"
img_input_ext = ".pgm"
pos_input_dir = "/idiap/group/vision/visidiap/databases/groundtruth/banca/english/eyecenter"
pos_input_ext = ".pos"
features_dir = os.path.join(base_output_TEMP_dir, "features_tantriggs")
features_ext = ".hdf5"
first_annot = 0
all_files_options = { }

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
