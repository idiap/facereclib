#!/usr/bin/env python

import facereclib
import bob

tool = facereclib.tools.LDATool

# PCA subspace; if not set, no PCA is performed
PCA_SUBSPACE_DIMENSION = 100
# LDA subspace; if not set, LDA subspace is not truncated
LDA_SUBSPACE_DIMENSION = 50

# the distance function to compare vectors in Fisher space
distance_function = bob.math.euclidean_distance
