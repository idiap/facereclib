#!/usr/bin/env python

import facereclib

#feature_extractor = facereclib.features.DCTBlocks
feature_extractor = facereclib.features.DCTBlocksVideo

# DCT blocks
BLOCK_H = 12
BLOCK_W = 12
OVERLAP_H = 11
OVERLAP_W = 11
N_DCT_COEF = 45

