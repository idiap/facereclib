#!/usr/bin/env python

import facereclib
import bob
import math

preprocessor = facereclib.preprocessing.FaceCrop

# Cropping
CROP_EYES_D = 33
CROP_H = 80
CROP_W = 64
CROP_OH = 16
CROP_OW = 32


feature_extractor = facereclib.features.GridGraph

# Gabor jet parameters
GABOR_DIRECTIONS = 8
GABOR_SCALES = 7
GABOR_SIGMA = math.sqrt(2.)*math.pi
GABOR_K_MAX = math.pi / 2.

normalize_jets = True
extract_phases = False


# tight grid grap paramaters
FIRST = (6, 6)
LAST = (74, 58)
STEP = (1, 1)

