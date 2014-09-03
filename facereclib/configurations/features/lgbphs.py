#!/usr/bin/env python

import facereclib
import math

# feature extraction
feature_extractor = facereclib.features.LGBPHS(
    # block setup
    block_size = 10,
    block_overlap = 4,
    # Gabor parameters
    gabor_sigma = math.sqrt(2.) * math.pi,
    # LBP setup (we use the defaults)

    # histogram setup
    sparse_histogram = True
)
