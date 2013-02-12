#!/usr/bin/env python

# setup of the grid parameters

# default queue used for training
training_queue = { 'queue':'q1d', 'memfree':'4G' }

# number of images that one job should preprocess
number_of_images_per_job = 1000
preprocessing_queue = {}

# number of features that one job should extract
number_of_features_per_job = 1000
extraction_queue = { 'queue':'q1d', 'memfree':'2G' }

# number of features that one job should project
number_of_projections_per_job = 50
projection_queue = { 'queue':'q1d', 'memfree':'2G' }

# not required
number_of_models_per_enroll_job = 10
enroll_queue = { 'queue':'q1d', 'memfree':'4G' }

# not required
number_of_models_per_score_job = 10
score_queue = { 'queue':'q1d', 'memfree':'4G' }
