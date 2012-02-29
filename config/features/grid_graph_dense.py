#!/usr/bin/env python

import facereclib
import bob
import math

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

feature_extractor = facereclib.features.GridGraph

# Gabor jet parameters
GABOR_SIGMA = math.sqrt(2.)*math.pi

normalize_jets = True
extract_phases = False


# tight grid grap paramaters
FIRST = (6, 6)
LAST = (74, 58)
STEP = (1, 1)

