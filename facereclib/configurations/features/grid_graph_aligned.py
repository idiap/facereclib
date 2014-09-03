#!/usr/bin/env python

import facereclib
import math

# copy the settings of the face_crop preprocessing
# here, we only need the eye positions
import os
exec(open(os.path.join(os.path.dirname(__file__), '..', 'preprocessing', 'face_crop.py')).read())

feature_extractor = facereclib.features.GridGraph(
    # Gabor parameters
    gabor_directions = 8,
    gabor_scales = 5,
    gabor_sigma = math.sqrt(2.) * math.pi,

    # what kind of information to extract
    normalize_gabor_jets = True,

    # setup of the aligned grid
    eyes = {'leye' : LEFT_EYE_POS, 'reye' : RIGHT_EYE_POS},
    nodes_between_eyes = 4,
    nodes_along_eyes = 2,
    nodes_above_eyes = 2,
    nodes_below_eyes = 7
)
