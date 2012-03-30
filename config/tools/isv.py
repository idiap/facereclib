#!/usr/bin/env python

import facereclib

tool = facereclib.tools.ISVTool

# 2/ GMM Training
n_gaussians = 512
iterk = 500
iterg_train = 500
end_acc = 0.0005
var_thd = 0.0005
update_weights = True
update_means = True
update_variances = True
norm_KMeans = True

# 3/ JFA Training
ru = 160 
relevance_factor = 4
n_iter_train = 10
n_iter_enrol = 1

# 4/ JFA Enrolment and scoring
iterg_enrol = 1
convergence_threshold = 0.0005
variance_threshold = 0.0005
responsibilities_threshold = 0


