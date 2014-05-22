#!/usr/bin/env python

import facereclib

# copy the settings of the UBM/GMM tool
import os
exec(open(os.path.join(os.path.dirname(__file__), 'plda.py')).read())

tool = facereclib.tools.PLDA(
    subspace_dimension_of_f = 16,  # Size of subspace F
    subspace_dimension_of_g = 16,  # Size of subspace G
    subspace_dimension_pca = 150   # Size of the PCA subspace
)
