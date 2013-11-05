#!/usr/bin/env python

import xbob.db.scface
import facereclib

scface_directory = "[YOUR_SC_FACE_DIRECTORY]"

# setup for SCface database
database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.scface.Database(),
    name = 'scface',
    original_directory = scface_directory,
    original_extension = ".jpg",
    has_internal_annotations = True,
    protocol = 'combined'
)
