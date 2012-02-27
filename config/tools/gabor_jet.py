#!/usr/bin/env python

import facereclib
import bob


# setup of the tool chain
tool = facereclib.tools.GaborJetTool

# extract average model?
extract_averaged_model = True

# Gabor jet comparison setup
jet_similarity_function = bob.machine.CanberraSimilarity()
