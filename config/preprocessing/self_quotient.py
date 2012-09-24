#!/usr/bin/env python

import facereclib

# copy the settings of the face cropping
import os
execfile(os.path.join(os.path.dirname(__file__), 'face_crop.py'))

preprocessor = facereclib.preprocessing.SelfQuotientImage

# self quotient image variance
VARIANCE = 2.
