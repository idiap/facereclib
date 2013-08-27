#!/usr/bin/env python

import xbob.db.scface
import facereclib

# setup for SCface database
database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.scface.Database(),
    name = 'scface',
    original_directory = "/idiap/group/biometric/databases/scface/images/",
    original_extension = ".jpg",
    has_internal_annotations = True,
    protocol = 'combined'
)
