#!/usr/bin/env python

import facereclib

tool = facereclib.tools.IVector(
    # IVector parameters
    subspace_dimension_of_t = 400,
    update_sigma = True,
    tv_training_iterations = 3,  # Number of EM iterations for the TV training
    # GMM parameters
    number_of_gaussians = 512,
    # by default, our features are normalized, so it does not need to be done here
    normalize_before_k_means = False
)
