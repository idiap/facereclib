#!/usr/bin/env python

import facereclib

preprocessor = facereclib.preprocessing.TanTriggs

# color channel 
color_channel = 'gray'

# Cropping
CROP_H = 80
CROP_W = CROP_H * 4 / 5
CROP_EYES_D = CROP_W / 2. + 1.
CROP_OH = CROP_H / 5.
CROP_OW = CROP_W / 2. - 0.5
OFFSET = 0

# Tan Triggs
GAMMA = 0.2
SIGMA0 = 1.
SIGMA1 = 2.
SIZE = 5
THRESHOLD = 10.
ALPHA = 0.1

