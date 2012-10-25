#!/usr/bin/env python

import facereclib

# copy the settings of the UBM/GMM tool
import os
execfile(os.path.join(os.path.dirname(__file__), 'plda.py'))

tool = facereclib.tools.PLDATool(
    subspace_dimension_of_f = 16,  # Size of subspace F
    subspace_dimension_of_g = 16,  # Size of subspace G
    subspace_dimension_pca = 200   # Size of the PCA subspace
)
