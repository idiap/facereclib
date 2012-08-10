#!/usr/bin/env python

import facereclib

tool = facereclib.tools.ISVVideoTool

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

# 3/ ISV Training
ru = 160 
relevance_factor = 4
n_iter_train = 10
n_iter_enrol = 1

# 4/ ISV Enrolment and scoring
iterg_enrol = 1
convergence_threshold = 0.0005
variance_threshold = 0.0005
responsibilities_threshold = 0

##############

frame_selector_for_train_projector  = facereclib.utils.video.FirstNFrameSelector(1) # Frames for UBM training
frame_selector_for_enrol            = facereclib.utils.video.AllFrameSelector()     # Frames for enrolment
frame_selector_for_project          = facereclib.utils.video.AllFrameSelector()     # Frames for scoring (via GMMStats)

##############


