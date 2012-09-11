#!/usr/bin/env python

import facereclib
import bob

tool = facereclib.tools.LDATool

PCA_SUBSPACE_DIMENSION = 100
LDA_SUBSPACE_DIMENSION = 50

distance_function = bob.math.euclidean_distance
