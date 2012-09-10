#!/usr/bin/env python

import xbob.db.banca

# 0/ The database to use
name = 'banca'
db = xbob.db.banca.Database()
protocol = 'Mc'

image_directory = "/idiap/group/vision/visidiap/databases/banca/english/images_gray/"
image_extension = ".pgm"
annotation_directory = "/idiap/group/vision/visidiap/databases/groundtruth/banca/english/eyecenter/"
annotation_type = 'eyecenter'

projector_training_options = { 'subworld': "twothirds" }
