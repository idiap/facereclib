#!/usr/bin/env python

import facereclib
import bob
import math

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
USE_PHASES = False
