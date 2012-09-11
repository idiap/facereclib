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

# eye positions for frontal images
RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT / 5, CROPPED_IMAGE_WIDTH / 4 - 1)
LEFT_EYE_POS  = (CROPPED_IMAGE_HEIGHT / 5, CROPPED_IMAGE_WIDTH / 4 * 3)

NODE_COUNT_BETWEEN_EYES = 4
NODE_COUNT_ALONG_EYES = 2
NODE_COUNT_ABOVE_EYES = 3
NODE_COUNT_BELOW_EYES = 7

