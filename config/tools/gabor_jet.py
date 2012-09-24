#!/usr/bin/env python

import facereclib
import bob
import math

# setup of the tool chain
tool = facereclib.tools.GaborJetTool

# extract average model?
EXTRACT_AVERAGED_MODELS = False


# copy the settings of the grid graph feature extraction; the Gabor parameters are needed to initialize the Gabor jet similarity
import os
execfile(os.path.join(os.path.dirname(__file__), '..', 'features', 'grid_graph.py'))

# Gabor jet comparison setup
GABOR_JET_SIMILARITY_TYPE = bob.machine.gabor_jet_similarity_type.PHASE_DIFF_PLUS_CANBERRA
