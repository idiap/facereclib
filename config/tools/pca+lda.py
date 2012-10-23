#!/usr/bin/env python

import facereclib
import bob

tool = facereclib.tools.LDATool(
    lda_subspace_dimension = 50,
    pca_subspace_dimension = 100,
    distance_function = bob.math.euclidean_distance
)
