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


# Grid graph parameters
# Cropping
CROPPED_IMAGE_HEIGHT = 80
CROPPED_IMAGE_WIDTH  = CROPPED_IMAGE_HEIGHT * 4 / 5

FIRST_NODE = (6, 6)
LAST_NODE = (CROPPED_IMAGE_HEIGHT - FIRST_NODE[0], CROPPED_IMAGE_WIDTH - FIRST_NODE[1])
NODE_DISTANCE = (4, 4)

