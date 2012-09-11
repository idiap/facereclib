#!/usr/bin/env python

import facereclib
import numpy

tool = facereclib.tools.BICTool

# Limit the number of training pairs
MAXIMUM_TRAINING_PAIR_COUNT = 10000

# Dimensions of intrapersonal and extrapersonal subspaces
INTRA_SUBSPACE_DIMENSION = 30
INTRA_SUBSPACE_DIMENSION = 30

# Distance measure to compare two features in image space
distance_function = numpy.subtract
USE_DFFS = False

