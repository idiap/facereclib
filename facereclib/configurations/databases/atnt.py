#!/usr/bin/env python

import xbob.db.atnt
import facereclib

database = facereclib.databases.DatabaseXBob(
    database = xbob.db.atnt.Database(),
    name = 'atnt',
    original_directory = "/idiap/group/biometric/databases/orl/",
    original_extension = ".pgm"
)
