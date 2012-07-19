#!/usr/bin/env python

import facereclib

preprocessor = facereclib.preprocessing.INormLBP


# LBP, always using 8 neighbors
RADIUS = 2
CIRCULAR = True
TO_AVERAGE = False
ADD_AVERAGE_BIT = False
UNIFORM = False
ROT_INV = False


# Cropping
CROP_H = 80
CROP_W = CROP_H * 4 / 5
CROP_EYES_D = CROP_W / 2 + 1
CROP_OH = CROP_H / 3
CROP_OW = CROP_W / 2
OFFSET = 0


