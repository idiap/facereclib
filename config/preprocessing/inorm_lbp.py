#!/usr/bin/env python

import facereclib

preprocessor = facereclib.preprocessing.INormLBP

# Cropping
CROP_H = 80
CROP_W = CROP_H * 4 / 5
CROP_EYES_D = CROP_W / 2 + 1
CROP_OH = CROP_H / 5
CROP_OW = CROP_W / 2

# LBP
RADIUS = 2
P_N = 8
CIRCULAR = True
TO_AVERAGE = False
ADD_AVERAGE_BIT = False
UNIFORM = False
ROT_INV = False

