#!/usr/bin/env python

import xbob.db.banca

# 0/ The database to use
name = 'banca'
db = xbob.db.banca.Database()
protocol = 'P'

img_input_dir = "/idiap/group/vision/visidiap/databases/banca/english/images_gray/"
img_input_ext = ".pgm"
pos_input_dir = "/idiap/group/vision/visidiap/databases/groundtruth/banca/english/eyecenter/"
pos_input_ext = ".pos"

annotation_type = 'eyecenter'

