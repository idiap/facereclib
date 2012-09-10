#!/usr/bin/env python

import facereclib

preprocessor = facereclib.preprocessing.TanTriggs

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

# Offset as will be required by the feature extraction -- here: the radius of the LBP
OFFSET = 2


# Tan Triggs
GAMMA = 0.2
SIGMA0 = 1.
SIGMA1 = 2.
SIZE = 5
THRESHOLD = 10.
ALPHA = 0.1

