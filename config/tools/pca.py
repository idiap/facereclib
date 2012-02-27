#!/usr/bin/env python

import facereclib
import bob

tool = facereclib.tools.PCATool

n_outputs = 300

distance_function = bob.math.euklidean_distance
