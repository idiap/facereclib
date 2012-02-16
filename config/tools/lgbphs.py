#!/usr/bin/env python

import tools
import bob
import math

tool = tools.LGBPHSTool
gabor_wavelet_transform = bob.ip.GaborWaveletTransform(5,8,math.sqrt(2.)*math.pi)


# LBP
BLOCK_H = 10
BLOCK_W = 10
OVERLAP_H = 4
OVERLAP_W = 4
RADIUS = 2
P_N = 8
CIRCULAR = True
TO_AVERAGE = False
ADD_AVERAGE_BIT = False
UNIFORM = True
ROT_INV = False


