#!/usr/bin/env python

import bob
import tools
import math

tool_chain = tools.LGBPHSToolChain
#gabor_wavelet_transform = bob.ip.GaborBankFrequency(80,64, eta=math.sqrt(2.), gamma=math.sqrt(2.))
gabor_wavelet_transform = bob.ip.GaborBankFrequency(80,64)


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


