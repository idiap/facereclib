#!/usr/bin/env python

import facereclib
import bob

tool = facereclib.tools.LGBPHS(
    distance_function = bob.math.histogram_intersection,
    is_distance_function = False
)

