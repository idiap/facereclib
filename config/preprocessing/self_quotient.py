#!/usr/bin/env python

import facereclib

preprocessor = facereclib.preprocessing.SelfQuotientImage

# color channel
COLOR_CHANNEL = 'gray'

# Cropping
CROPPED_IMAGE_HEIGHT = 80
CROPPED_IMAGE_WIDTH  = CROPPED_IMAGE_HEIGHT * 4 / 5

# eye positions for frontal images
RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT / 5, CROPPED_IMAGE_WIDTH / 4 - 1)
LEFT_EYE_POS  = (CROPPED_IMAGE_HEIGHT / 5, CROPPED_IMAGE_WIDTH / 4 * 3)

# eye and mouth position for profile images
# (only appropriate for left profile images; change them for right profiles)
EYE_POS = (16, 20)
MOUTH_POS = (52, 20)

# Offset as will be required by the feature extraction -- here: None
OFFSET = 0


# self quotient image
VARIANCE = 2.
