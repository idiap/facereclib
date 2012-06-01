#!/usr/bin/env python

import bob

# 0/ The database to use
name = 'banca'
db = bob.db.banca.Database()
protocol = 'P'

img_input_dir = "/idiap/group/vision/visidiap/databases/banca/english/images_gray/"
img_input_ext = ".pgm"
pos_input_dir = "/idiap/group/vision/visidiap/databases/groundtruth/banca/english/eyecenter/"
pos_input_ext = ".pos"

first_annot = 0
world_projector_options = { 'subworld': "twothirds" }

