#!/usr/bin/env python

import facereclib
import math

feature_extractor = facereclib.features.GridGraph

# Gabor jet parameters
GABOR_DIRECTIONS = 8
GABOR_SCALES = 5
GABOR_SIGMA = math.sqrt(2.) * math.pi
GABOR_MAXIMUM_FREQUENCY = math.pi / 2.
GABOR_FREQUENCY_STEP = math.sqrt(.5)
GABOR_POWER_OF_K = 0
GABOR_DC_FREE = True

NORMALIZE_GABOR_JETS = True
EXTRACT_GABOR_PHASES = True


# copy the settings of the face_crop preprocessing; they are needed to limit the grid
import os
execfile(os.path.join(os.path.dirname(__file__), '..', 'preprocessing', 'face_crop.py'))

# aligned grid graph parameters
NODE_COUNT_BETWEEN_EYES = 4
NODE_COUNT_ALONG_EYES = 2
NODE_COUNT_ABOVE_EYES = 3
NODE_COUNT_BELOW_EYES = 7

