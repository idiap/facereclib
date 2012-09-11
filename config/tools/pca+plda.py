#!/usr/bin/env python

import facereclib

# copy the settings of the UBM/GMM tool
import os
execfile(os.path.join(os.path.dirname(__file__), 'plda.py'))

tool = facereclib.tools.PLDATool

# PCA subspace
SUBSPACE_DIMENSION_PCA = 200

