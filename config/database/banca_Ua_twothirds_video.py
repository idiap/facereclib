#!/usr/bin/env python

import xbob.db.banca
import facereclib

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.banca.Database(),
    name = "banca",
    image_directory = "/idiap/home/rwallace/work/databases/banca-video/output/frames/", # hdf5 files, each containing cropped frames
    image_extension = ".hdf5",
    annotation_directory = "/idiap/group/vision/visidiap/databases/groundtruth/banca/english/eyecenter/",
    annotation_type = 'eyecenter',
    protocol = 'Ua',
    projector_training_options = { 'subworld': "twothirds" }
)
