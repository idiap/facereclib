#!/usr/bin/env python

import facereclib

tool = facereclib.tools.ISVVideo(
    # frame selectors
    frame_selector_for_projector_training = facereclib.utils.video.FirstNFrameSelector(1), # Frames for UBM training
    frame_selector_for_enroll             = facereclib.utils.video.AllFrameSelector(),     # Frames for enrollment
    frame_selector_for_projection         = facereclib.utils.video.AllFrameSelector(),     # Frames for scoring (via GMMStats)
    subspace_dimension_of_u = 160,
    number_of_gaussians = 512
)
