#!/usr/bin/env python

import facereclib
import bob
import math

# setup of the tool chain
tool = facereclib.tools.GaborJets(
    # Gabor jet comparison
    gabor_jet_similarity_type = bob.machine.gabor_jet_similarity_type.PHASE_DIFF_PLUS_CANBERRA,
    # Gabor wavelet setup
    gabor_sigma = math.sqrt(2.) * math.pi
)
