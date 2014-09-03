#!/usr/bin/env python

import bob.db.arface
import facereclib

arface_directory = "[YOUR_ARFACE_DIRECTORY]"

database = facereclib.databases.DatabaseXBob(
    database = bob.db.arface.Database(
      original_directory = arface_directory
    ),
    name = 'arface',
    original_directory = arface_directory,
    original_extension = ".ppm",
    has_internal_annotations = True,
    protocol = 'all'
)
