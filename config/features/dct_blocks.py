#!/usr/bin/env python

import facereclib

feature_extractor = facereclib.features.DCTBlocks

# blocks
BLOCK_HEIGHT = 12
BLOCK_WIDTH = 12
BLOCK_Y_OVERLAP = 11
BLOCK_X_OVERLAP = 11

# DCT setup
NUMBER_OF_DCT_COEFFICIENTS = 45
