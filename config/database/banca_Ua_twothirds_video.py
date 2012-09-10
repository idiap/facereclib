#!/usr/bin/env python

import xbob.db.banca

# 0/ The database to use
name = 'banca'
db = xbob.db.banca.Database()
protocol = 'Ua'

image_directory = "/idiap/home/rwallace/work/databases/banca-video/output/frames/" # hdf5 files, each containing cropped frames
image_extension = ".hdf5"

projector_training_options = { 'subworld': "twothirds" }

