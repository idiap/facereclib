#!/usr/bin/env python

import xbob.db.banca

# 0/ The database to use
name = 'banca'
db = xbob.db.banca.Database()
protocol = 'Mc'

img_input_dir = "/idiap/home/rwallace/work/databases/banca-video/output/frames/" # hdf5 files, each containing cropped frames
img_input_ext = ".hdf5"
pos_input_dir = None # None because hdf5 files in img_input_dir are pre-cropped with Omniperception annotations
pos_input_ext = None

world_projector_options = { 'subworld': "twothirds" }

