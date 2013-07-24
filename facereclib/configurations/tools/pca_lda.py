#!/usr/bin/env python

import facereclib
import scipy.spatial

tool = facereclib.tools.LDA(
    lda_subspace_dimension = 50,
    pca_subspace_dimension = 100,
    distance_function = scipy.spatial.distance.euclidean,
    is_distance_function = True
)
