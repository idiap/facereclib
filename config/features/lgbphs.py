#!/usr/bin/env python

import facereclib
import bob
import math

# feature extraction
feature_extractor = facereclib.features.LGBPHS

# Block setup
BLOCK_HEIGHT = 10
BLOCK_WIDTH = 10
BLOCK_Y_OVERLAP = 4
BLOCK_X_OVERLAP = 4


# LBP parameters
RADIUS = 2
NEIGHBOR_COUNT = 8
IS_UNIFORM = True
IS_CIRCULAR = True
IS_ROTATION_INVARIANT = False
COMPARE_TO_AVERAGE = False
ADD_AVERAGE_BIT = False


# Gabor parameters
GABOR_DIRECTIONS = 8
GABOR_SCALES = 5
GABOR_SIGMA = math.sqrt(2.) * math.pi
GABOR_MAXIMUM_FREQUENCY = math.pi / 2.
GABOR_FREQUENCY_STEP = math.sqrt(.5)
GABOR_POWER_OF_K = 0
GABOR_DC_FREE = True


# How to split the histogram
USE_SPARSE_HISTOGRAM = True
SPLIT_HISTOGRAM = None
USE_GABOR_PHASES = False
