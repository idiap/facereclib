#!/usr/bin/env python

import preprocessing
import features
import bob
import math


preprocessor = preprocessing.FaceCrop

# Cropping
CROP_EYES_D = 33
CROP_H = 80
CROP_W = 64
CROP_OH = 16
CROP_OW = 32


feature_extractor = features.GridGraph

# Gabor jet parameters
GABOR_SIGMA = math.sqrt(2.)*math.pi
GABOR_K_MAX = math.pi / 2.

normalize_jets = True
extract_phases = True


# 2/ Grid graph parameters
COUNT_BETWEEN_EYES = 3
COUNT_ALONG_EYES = 1
COUNT_ABOVE_EYES = 1
COUNT_BELOW_EYES = 7

