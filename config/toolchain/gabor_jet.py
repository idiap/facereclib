#!/usr/bin/env python

import os
import bob
import toolchain


# setup of the tool chain
tool_chain = toolchain.GaborJets.GaborJetToolChain
gabor_wavelet_transform = bob.ip.GaborWaveletTransform()

# 2/ Compute Gabor graphs
LEFT_EYE_POSITION = [49, 16]
RIGHT_EYE_POSITION = [16, 16]
COUNT_BETWEEN_EYES = 3
COUNT_ALONG_EYES = 1
COUNT_ABOVE_EYES = 2
COUNT_BELOW_EYES = 4

# extract average model?
extract_averaged_model = False

# Gabor jet comparison setup
jet_similarity_function = bob.machine.CanberraSimilarity()


