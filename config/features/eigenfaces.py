#!/usr/bin/env python

import facereclib

preprocessor = facereclib.preprocessing.TanTriggs

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



# compute eigenfaces using the training database
feature_extractor = facereclib.features.Eigenface

n_outputs = 300

