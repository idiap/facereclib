#!/usr/bin/env python

import facereclib
import numpy

feature_extractor = facereclib.features.SIFTBobKeypoints(
    sigmas = [1.75, 3.06, 9.38], 
    height = 200,
    width = 200,
    n_octaves = 5,
    n_scales = 3,
    octave_min = 0,
    sigma_n = 0.5,
    sigma0 = 1.6,
    contrast_thres = 0.03,
    edge_thres = 10.,
    norm_thres = 0.2,
    kernel_radius_factor = 4.,
    set_sigma0_no_init_smoothing = True,
)
