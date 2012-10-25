#!/usr/bin/env python

import facereclib
import bob

tool = facereclib.tools.LDATool(
    lda_subspace_dimension = 50,
    distance_function = bob.math.euclidean_distance
)
