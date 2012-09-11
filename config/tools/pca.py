#!/usr/bin/env python

import facereclib
import bob

tool = facereclib.tools.PCATool

SUBSPACE_DIMENSION = 300

distance_function = bob.math.euclidean_distance
