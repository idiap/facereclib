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

#gabor_wavelet_transform = bob.ip.GaborWaveletTransform(5,8,sigma = math.sqrt(2.) * math.pi, k_max = math.pi)
gabor_wavelet_transform = bob.ip.GaborWaveletTransform(7,8,sigma = math.sqrt(2.) * math.pi, k_max = math.pi / math.sqrt(2.))
normalize_jets = True
extract_phases = False

# tight grid grap paramaters
FIRST = (6, 6)
LAST = (74, 58)
STEP = (1, 1)

