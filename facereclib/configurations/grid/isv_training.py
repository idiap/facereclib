#!/usr/bin/env python

# setup of the grid parameters

# default queue used for training
training_queue = { 'queue':'q1d', 'memfree':'4G', 'io_big':True }

# the queue that is used solely for the final ISV training step
isv_training_queue = { 'queue':'q1wm', 'memfree':'32G', 'pe_opt':'pe_mth 4', 'hvmem':'8G' }

# number of images that one job should preprocess
number_of_images_per_job = 1000
preprocessing_queue = {}

# number of features that one job should extract
number_of_features_per_job = 1000
extraction_queue = { 'queue':'q1d', 'memfree':'2G' }

# number of features that one job should project
number_of_projections_per_job = 200
projection_queue = { 'queue':'q1d', 'memfree':'2G' }

# number of models that one job should enroll
number_of_models_per_enroll_job = 20
enroll_queue = { 'queue':'q1d', 'memfree':'4G', 'io_big':True }

# number of models that one score job should process
number_of_models_per_score_job = 20
score_queue = { 'queue':'q1d', 'memfree':'4G', 'io_big':True }
