#!/usr/bin/env python

import facereclib

# copy the settings of the face cropping with offset
import os
execfile(os.path.join(os.path.dirname(__file__), 'face_crop_with_offset.py'))

# copy the settings of the Tan & Triggs algorithm
execfile(os.path.join(os.path.dirname(__file__), 'tan_triggs.py'))

preprocessor = facereclib.preprocessing.TanTriggs

