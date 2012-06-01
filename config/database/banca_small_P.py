#!/usr/bin/env python

import bob

# Note: if using with faceverify_zt.py, set options "--groups dev" (there is no test set) and "--no-zt-norm" (doesn't work otherwise)
# 0/ The database to use
name = 'banca_small'
db = bob.db.banca_small.Database()
protocol = 'P'

img_input_dir = "/idiap/group/vision/visidiap/databases/banca/english/images_gray/"
img_input_ext = ".pgm"
pos_input_dir = "/idiap/group/vision/visidiap/databases/groundtruth/banca/english/eyecenter/"
pos_input_ext = ".pos"

first_annot = 0

