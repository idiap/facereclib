#!/usr/bin/env python

import facereclib
import math

# copy the settings of the face_crop preprocessing;
# here, we need the image resolution
import os
execfile(os.path.join(os.path.dirname(__file__), '..', 'preprocessing', 'face_crop.py'))

feature_extractor = facereclib.features.GridGraph(
    # Gabor parameters
    gabor_sigma = math.sqrt(2.) * math.pi,

    # what kind of information to extract
    normalize_gabor_jets = True,
    extract_gabor_phases = True,

    # setup of the fixed grid
    first_node = (6, 6),
    last_node = (CROPPED_IMAGE_HEIGHT - 6, CROPPED_IMAGE_WIDTH - 6),
    node_distance = (4, 4)
)

