#!/usr/bin/env python

import facereclib

# copy the settings of the face cropping
import os
execfile(os.path.join(os.path.dirname(__file__), 'face_crop.py'))

preprocessor = facereclib.preprocessing.INormLBP

# LBP, always using 8 neighbors
RADIUS = 2
IS_UNIFORM = False
IS_CIRCULAR = True
IS_ROTATION_INVARIANT = False
COMPARE_TO_AVERAGE = False
ADD_AVERAGE_BIT = False
