#!/usr/bin/env python

import bob

# 0/ The database to use
name = 'banca'
db = bob.db.banca.Database()
protocol = 'Mc'

#img_input_dir = "/idiap/group/vision/visidiap/databases/banca/english/images_gray/"
#img_input_ext = ".pgm"
img_input_dir = "/idiap/home/rwallace/work/databases/banca-video/output/frames/" # hdf5 files, each containing cropped frames
img_input_ext = ".hdf5"
#pos_input_dir = "/idiap/group/vision/visidiap/databases/groundtruth/banca/english/eyecenter/"
#pos_input_ext = ".pos"
pos_input_dir = None # None because hdf5 files in img_input_dir are pre-cropped with Omniperception annotations
pos_input_ext = None

first_annot = 0
all_files_options = {}
world_extractor_options = {}
world_projector_options = { 'subworld': "twothirds" }
world_enroler_options = {}
