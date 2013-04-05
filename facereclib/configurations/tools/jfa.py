#!/usr/bin/env python

import facereclib

tool = facereclib.tools.JFA(
    # JFA Training
    subspace_dimension_of_u = 2, # U subspace dimension
    subspace_dimension_of_v = 2, # V subspace dimension
    jfa_training_iterations = 10, # Number of EM iterations for the JFA training
    # GMM training
    number_of_gaussians = 512
)
