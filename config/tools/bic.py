#!/usr/bin/env python

import facereclib
import numpy

tool = facereclib.tools.BICTool

# Limit the number of training pairs
maximum_pair_count = 10000

# Dimensions of intrapersonal and extrapersonal subspaces
M_I = 30
M_E = 30

# Distance measure to compare two features in image space
distance_function = numpy.subtract
USE_DFFS = False

