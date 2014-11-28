#!/usr/bin/env python

import facereclib
import numpy

tool = facereclib.tools.BIC(
    # Distance measure to compare two features in input space
    comparison_function = numpy.subtract,
    # Limit the number of training pairs
    maximum_training_pair_count = 10000,
    # Dimensions of intrapersonal and extrapersonal subspaces
    subspace_dimensions = (30, 30)
)
