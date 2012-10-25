#!/usr/bin/env python

import facereclib

# compute eigenfaces using the training database
feature_extractor = facereclib.features.Eigenface(
    subspace_dimension = 100
)
