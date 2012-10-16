#!/usr/bin/env python

import xbob.db.banca
import facereclib

database = facereclib.databases.DatabaseXBob(
  database = xbob.db.banca.Database(),
  name = "banca",
  image_directory = "/idiap/group/vision/visidiap/databases/banca/english/images_gray/",
  image_extension = ".pgm",
  annotation_directory = "/idiap/group/vision/visidiap/databases/groundtruth/banca/english/eyecenter/",
  annotation_type = 'eyecenter',
  protocol = 'P'
)


