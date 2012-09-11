#!/usr/bin/env python

import facereclib
import bob

tool = facereclib.tools.LGBPHSTool

# distance function
distance_function = bob.math.histogram_intersection
IS_DISTANCE_FUNCTION = False

