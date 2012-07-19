#!/usr/bin/env python

import facereclib

preprocessor = facereclib.preprocessing.SelfQuotientImage

# color channel 
color_channel = 'gray'


# Cropping
CROP_H = 80
CROP_W = CROP_H * 4 / 5
CROP_EYES_D = CROP_W / 2 + 1
CROP_OH = CROP_H / 3
CROP_OW = CROP_W / 2
OFFSET = 0


# self quotient image
size = 10
sigma = 5.
