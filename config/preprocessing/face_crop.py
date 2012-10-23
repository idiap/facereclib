#!/usr/bin/env python

import facereclib

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

# define the preprocessor
preprocessor = facereclib.preprocessing.FaceCrop(
    cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
    cropped_positions = {'leye' : LEFT_EYE_POS, 'reye' : RIGHT_EYE_POS, 'eye' : EYE_POS, 'mouth' : MOUTH_POS}
)
