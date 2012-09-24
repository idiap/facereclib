#!/usr/bin/env python

import facereclib

# copy the settings of the Tan & Triggs algorithm
import os
execfile(os.path.join(os.path.dirname(__file__), 'tan_triggs.py'))

preprocessor = facereclib.preprocessing.TanTriggsVideo
