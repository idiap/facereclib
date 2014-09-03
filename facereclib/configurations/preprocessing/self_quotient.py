#!/usr/bin/env python

import facereclib
import math

# copy the settings of the face cropping
import os
exec(open(os.path.join(os.path.dirname(__file__), 'face_crop.py')).read())

# we use the Self quotient image technology with variance 2.
preprocessor = facereclib.preprocessing.SelfQuotientImage(
    cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
    cropped_positions = {'leye' : LEFT_EYE_POS, 'reye' : RIGHT_EYE_POS, 'eye' : EYE_POS, 'mouth' : MOUTH_POS},
    sigma = math.sqrt(2.)
)
