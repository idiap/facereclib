#!/usr/bin/env python

import facereclib

# copy the settings of the face cropping
import os
execfile(os.path.join(os.path.dirname(__file__), 'face_crop.py'))

preprocessor = facereclib.preprocessing.TanTriggs

# Tan Triggs
GAMMA = 0.2
SIGMA0 = 1.
SIGMA1 = 2.
SIZE = 5
THRESHOLD = 10.
ALPHA = 0.1

