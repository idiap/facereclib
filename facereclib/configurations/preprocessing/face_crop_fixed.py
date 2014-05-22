#!/usr/bin/env python

import facereclib

# copy the settings of the face cropping
import os
exec(open(os.path.join(os.path.dirname(__file__), 'face_crop.py')).read())

# fixed locations of the eyes; if this is set, the hand-labeled eye positions will be ignored (if available),
# and instead it is assumed that a detector was run before and put the eye positions to these locations:
# Here, I have put positions that should fit well with the LFW database
FIXED_RIGHT_EYE_POS = (110, 100)
FIXED_LEFT_EYE_POS = (110, 150)

preprocessor = facereclib.preprocessing.FaceCrop(
    cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
    cropped_positions = {'leye' : LEFT_EYE_POS, 'reye' : RIGHT_EYE_POS, 'eye' : EYE_POS, 'mouth' : MOUTH_POS},
    fixed_positions = {'leye' : FIXED_LEFT_EYE_POS, 'reye' : FIXED_RIGHT_EYE_POS}
)
