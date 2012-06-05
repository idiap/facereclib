#!/usr/bin/env python

import facereclib
import bob

tool = facereclib.tools.UBMGMMVideoTool


# 2/ GMM Training
n_gaussians = 512
iterk = 500
iterg_train = 500
update_weights = True
update_means = True
update_variances = True
norm_KMeans = True

##############
frame_selector_for_train = facereclib.frame_selection.AllFrameSelector()

# use only the first N frames from each video for UBM training
#frame_selector_for_train = facereclib.frame_selection.FirstNFrameSelector(5)
##############

# 3/ GMM Enrolment and scoring
iterg_enrol = 1
convergence_threshold = 0.0005
variance_threshold = 0.0005
relevance_factor = 4
responsibilities_threshold = 0

# Scoring
scoring_function = bob.machine.linear_scoring

