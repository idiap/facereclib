#!/usr/bin/env python

import facereclib
import bob

tool = facereclib.tools.LDATool

pca_subspace = 100
lda_subspace = 50

distance_function = bob.math.euclidean_distance
