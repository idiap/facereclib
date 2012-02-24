#!/usr/bin/env python

import preprocessing
import features
import bob
import math

preprocessor = preprocessing.TanTriggs

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



# feature extraction
feature_extractor = features.LGBPHS

# Gabor parameters
SIGMA_GABOR = math.sqrt(2.)*math.pi
KMAX_GABOR = math.pi / 2.


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


