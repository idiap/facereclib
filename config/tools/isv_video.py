#!/usr/bin/env python

import facereclib

# copy the settings of the ISV tool
import os
execfile(os.path.join(os.path.dirname(__file__), 'isv.py'))

tool = facereclib.tools.ISVVideoTool

# frame selectors

frame_selector_for_projector_training = facereclib.utils.video.FirstNFrameSelector(1) # Frames for UBM training
frame_selector_for_enroll             = facereclib.utils.video.AllFrameSelector()     # Frames for enrollment
frame_selector_for_projection         = facereclib.utils.video.AllFrameSelector()     # Frames for scoring (via GMMStats)


