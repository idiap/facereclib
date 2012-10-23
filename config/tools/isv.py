#!/usr/bin/env python

import facereclib

tool = facereclib.tools.ISVTool(
    # ISV parameters
    subspace_dimension_of_u = 160,
    # GMM parameters
    number_of_gaussians = 512
)
