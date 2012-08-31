#!/usr/bin/env python

import facereclib

feature_extractor = facereclib.features.SIFTKeypoints

# SIFT parameters
N_SCALES = 1 # Maximum: 5 scales
SIGMA0 = 3
#SIGMA1 = 0.0625
#SIGMA2 = 0.0277778
#SIGMA3 = 0.0277778
#SIGMA4 = 0.0277778
HEIGHT = 250
WIDTH = 250

N_INTERVALS = 3
N_OCTAVES = 5
OCTAVE_MIN = 0
EDGE_THRES = 10.
PEAK_THRES = 0.03
MAGNIF = 3.

ESTIMATE_ORIENTATION = False
