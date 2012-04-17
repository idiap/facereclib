#!/usr/bin/env python

import facereclib
import bob
import math

preprocessor = facereclib.preprocessing.TanTriggs

# Cropping
CROP_H = 80
CROP_W = CROP_H / 5 * 4
CROP_EYES_D = CROP_W / 2 + 1
CROP_OH = CROP_H / 5
CROP_OW = CROP_W / 2

# Tan Triggs
GAMMA = 0.2
SIGMA0 = 1.
SIGMA1 = 2.
SIZE = 5
THRESHOLD = 10.
ALPHA = 0.1



# feature extraction
feature_extractor = facereclib.features.LGBPHS

# Gabor parameters
GABOR_DIRECTIONS = 8
GABOR_SCALES = 5
GABOR_SIGMA = math.sqrt(2.) * math.pi
GABOR_K_MAX = math.pi / 2.
GABOR_K_FAC = math.sqrt(.5)
GABOR_POW_OF_K = 0
GABOR_DC_FREE = True


# LBP
BLOCK_H = 10
BLOCK_W = 10
OVERLAP_H = 4
OVERLAP_W = 4
RADIUS = 2
P_N = 8
CIRCULAR = True
TO_AVERAGE = False
ADD_AVERAGE_BIT = False
UNIFORM = True
ROT_INV = False

# How to split the histogram
SPLIT = None
