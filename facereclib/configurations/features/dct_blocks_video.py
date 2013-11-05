#!/usr/bin/env python

import facereclib

feature_extractor = facereclib.features.DCTBlocksVideo(
    block_size = 12,
    block_overlap = 11,
    number_of_dct_coefficients = 45
)
