#!/usr/bin/env python

import facereclib
import bob
import math

feature_extractor = facereclib.features.GridGraph

# Gabor jet parameters
GABOR_DIRECTIONS = 8
GABOR_SCALES = 5
GABOR_SIGMA = math.sqrt(2.) * math.pi
GABOR_K_MAX = math.pi / 2.
GABOR_K_FAC = math.sqrt(.5)
GABOR_POW_OF_K = 0
GABOR_DC_FREE = True

normalize_jets = True
extract_phases = True


# 2/ Grid graph parameters
CROP_H = 80
CROP_W = CROP_H * 4 / 5
CROP_EYES_D = CROP_W / 2 + 1
CROP_OH = CROP_H / 3
CROP_OW = CROP_W / 2

COUNT_BETWEEN_EYES = 4
COUNT_ALONG_EYES = 2
COUNT_ABOVE_EYES = 3
COUNT_BELOW_EYES = 7

