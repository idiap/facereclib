#!/usr/bin/env python

import facereclib
import bob

tool = facereclib.tools.UBMGMMTool

# GMM Training
GAUSSIANS = 512
K_MEANS_TRAINING_ITERATIONS = 500 # Number of iterations for K-Means
GMM_TRAINING_ITERATIONS = 500 # Number of iterations for ML GMM Training
GMM_TRAINING_THRESHOLD = 0.0005 # Threshold to end the ML training
GMM_VARIANCE_THRESHOLD = 0.0005 # Minimum value that a variance can reach
UPDATE_WEIGTHS = True
UPDATE_MEANS = True
UPDATE_VARIANCES = True
NORMALIZE_BEFORE_K_MEANS = True # Normalize the input features before running K-Means


# GMM Enrollment and scoring
RELEVANCE_FACTOR = 4 # Relevance factor as described in Reynolds paper
GMM_ENROLL_ITERATIONS = 1 # Number of iterations for the enrollment phase
RESPONSIBILITY_THRESHOLD = 0 # If set, the weight of a particular Gaussian will at least be greater than this threshold. In the case the real weight is lower, the prior mean value will be used to estimate the current mean and variance.

scoring_function = bob.machine.linear_scoring

