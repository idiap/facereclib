#!/usr/bin/env python

import facereclib
import math

# setup of the tool chain
tool = facereclib.tools.GaborJets(
    # Gabor jet comparison
    gabor_jet_similarity_type = "PhaseDiffPlusCanberra",
    # Gabor wavelet setup
    gabor_sigma = math.sqrt(2.) * math.pi
)
