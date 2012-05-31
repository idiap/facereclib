#!/usr/bin/env python

import facereclib

#preprocessor = facereclib.preprocessing.TanTriggs
preprocessor = facereclib.preprocessing.TanTriggsVideo

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

#feature_extractor = facereclib.features.DCTBlocks
feature_extractor = facereclib.features.DCTBlocksVideo

# DCT blocks
BLOCK_H = 12
BLOCK_W = 12
OVERLAP_H = 11
OVERLAP_W = 11
N_DCT_COEF = 45

