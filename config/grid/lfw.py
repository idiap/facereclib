#!/usr/bin/env python

# setup of the grid parameters

# default queue used for training
training_queue = { 'queue':'q1dm', 'memfree':'16G', 'pe_opt': 'pe_mth 2', 'hvmem' : '8G' }

# number of images that one job should preprocess
number_of_images_per_job = 1000
preprocessing_queue = {}

# number of features that one job should extract
number_of_features_per_job = 1000
extraction_queue = {}

# number of features that one job should project
number_of_projections_per_job = 1000
projection_queue = {}

# number of models that should be enroled by one enrol job
number_of_models_per_enrol_job = 1000
enrol_queue = { 'queue':'q1d', 'memfree':'8G' }

# number of models that one score computation should use
number_of_models_per_score_job = 1000
score_queue = { 'queue':'q1d', 'memfree':'8G' }
