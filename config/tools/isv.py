#!/usr/bin/env python

import facereclib

tool = facereclib.tools.ISVTool

# 2/ GMM Training
n_gaussians = 512
iterk = 500 # Number of iterations for K-Means
iterg_train = 500 # Number of iterations for ML GMM Training
end_acc = 0.0005 # Threshold to end the ML training
var_thd = 0.0005 # Minimum value that a variance can reach
update_weights = True
update_means = True
update_variances = True
norm_KMeans = True # Normalize the input features before running K-Means

# 3/ JFA Training
ru = 160 # U subspace dimension
relevance_factor = 4 # Relevance factor as described in Reynolds paper
n_iter_train = 10 # Number of EM iterations for the JFA training
n_iter_enroll = 1 # Number of iterations for the enrollment phase

# 4/ JFA Enrolment and scoring
convergence_threshold = 0.0005 # Same as for GMM
variance_threshold = 0.0005 # Same as for GMM
responsibilities_threshold = 0 # If set, the weight of a particular Gaussian will at least be greater than this threshold. In the case the real weight is lower, the prior mean value will be used to estimate the current mean and variance.


