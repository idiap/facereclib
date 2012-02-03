#!/usr/bin/env python

import os
import bob
import toolchain

tool_chain = toolchain.LGBPHS.LGBPHSToolChain
gabor_wavelet_transform = bob.ip.GaborWaveletTransform()


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


