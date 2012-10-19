#!/usr/bin/env python

import facereclib

preprocessor = facereclib.preprocessing.FaceCrop

# color channel
COLOR_CHANNEL = 'gray'

# Cropping
CROPPED_IMAGE_HEIGHT = 80
CROPPED_IMAGE_WIDTH  = CROPPED_IMAGE_HEIGHT * 4 / 5

# eye positions for frontal images
RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT / 5, CROPPED_IMAGE_WIDTH / 4 - 1)
LEFT_EYE_POS  = (CROPPED_IMAGE_HEIGHT / 5, CROPPED_IMAGE_WIDTH / 4 * 3)

# fixed locations of the eyes; if this is set, the hand-labeled eye positions will be ignored (if available),
# and instead it is assumed that a detector was run before and put the eye positions to these locations:
# Here, I have put positions that should fit well with the LFW database
FIXED_RIGHT_EYE = (110, 100)
FIXED_LEFT_EYE = (110, 150)

# Offset as will be required by the feature extraction -- here: None
OFFSET = 0
