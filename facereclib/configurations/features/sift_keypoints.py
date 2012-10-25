#!/usr/bin/env python

import facereclib

feature_extractor = facereclib.features.SIFTKeypoints(
    sigmas = (3,), # (3, 0.0625, 0.0277778, 0.0277778, 0.0277778)
    height = 250,
    width = 250,
    n_intervals = 3,
    n_octaves = 5,
    octave_min = 0,
    edge_thres = 10,
    peak_thres = 0.03,
    magnif = 3.,
    estimate_orientation = False
)
