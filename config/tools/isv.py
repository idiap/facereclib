#!/usr/bin/env python

import facereclib

# copy the settings of the UBM/GMM tool
import os
execfile(os.path.join(os.path.dirname(__file__), 'ubm_gmm.py'))

tool = facereclib.tools.ISVTool

# JFA Training
SUBSPACE_DIMENSION_OF_U = 160 # U subspace dimension
JFA_TRAINING_ITERATIONS = 10 # Number of EM iterations for the JFA training
JFA_TRAINING_THRESHOLD = 0.0005 # Same as for GMM

# JFA Enrollment and scoring
JFA_ENROLL_ITERATIONS = 1 # Number of iterations for the enrollment phase
JFA_VARIANCE_THRESHOLD = 0.0005 # Same as for GMM
