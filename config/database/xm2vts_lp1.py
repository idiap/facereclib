#!/usr/bin/env python

import xbob.db.xm2vts
import facereclib

# setup for XM2VTS
database = facereclib.databases.DatabaseXBob(
  database = xbob.db.xm2vts.Database(),
  name = "xm2vts",
  image_directory = "/idiap/resource/database/xm2vtsdb/images/",
  image_extension = ".ppm",
  annotation_directory = "/idiap/user/mguenther/annotations/XM2VTS/",
  annotation_type = 'eyecenter',
  protocol = 'lp1'
)
