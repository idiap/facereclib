#!/usr/bin/env python

import facereclib
import math

# copy the settings of the face_crop preprocessing;
# here, we need the image resolution
import os
exec(open(os.path.join(os.path.dirname(__file__), '..', 'preprocessing', 'face_crop.py')).read())

feature_extractor = facereclib.features.GridGraph(
    # Gabor parameters
    gabor_sigma = math.sqrt(2.) * math.pi,

    # what kind of information to extract
    normalize_gabor_jets = True,

    # setup of the fixed grid
    node_distance = (4, 4),
    first_node = (6, 6),
    image_resolution = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH)
)

