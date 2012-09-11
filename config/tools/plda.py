#!/usr/bin/env python

import facereclib

tool = facereclib.tools.PLDATool

PLDA_TRAINING_ITERATIONS = 200 # Maximum number of iterations for the EM loop
PLDA_TRAINING_THRESHOLD = 1e-3 # Threshold for ending the EM loop

SUBSPACE_DIMENSION_OF_F = 16 # Size of subspace F
SUBSPACE_DIMENSION_OF_G = 16 # Size of subspace G

INIT_SEED = 0 # seed for initializing
INIT_F_METHOD = 1
INIT_F_RATIO = 1
INIT_G_METHOD = 1
INIT_G_RATIO = 1
INIT_S_METHOD = 3 #
INIT_S_RATIO = 1
