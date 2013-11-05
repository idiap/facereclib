#!/usr/bin/env python

import facereclib
import scipy.spatial

tool = facereclib.tools.PCA(
    subspace_dimension = 30,
    distance_function = scipy.spatial.distance.euclidean,
    is_distance_function = True
)

