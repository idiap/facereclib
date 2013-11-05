#!/usr/bin/env python

import xbob.db.arface
import facereclib

arface_directory = "[YOUR_ARFACE_DIRECTORY]"

database = facereclib.databases.DatabaseXBob(
    database = xbob.db.arface.Database(),
    name = 'arface',
    original_directory = arface_directory,
    original_extension = ".ppm",
    has_internal_annotations = True,
    protocol = 'all'
)
