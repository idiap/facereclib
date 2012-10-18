#!/usr/bin/env python

import xbob.db.atnt
import facereclib

database = facereclib.databases.DatabaseXBob(
  database = xbob.db.atnt.Database(),
  name = 'atnt',
  image_directory = "/idiap/group/biometric/databases/orl/",
  image_extension = ".pgm"
)
