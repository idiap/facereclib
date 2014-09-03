#!/usr/bin/env python

import bob.db.frgc
import facereclib

frgc_directory = "[YOUR_FRGC_DIRECTORY]"

database = facereclib.databases.DatabaseXBob(
    database = bob.db.frgc.Database(frgc_directory),
    name = "frgc",
    original_directory = frgc_directory,
    original_extension = ".jpg",
    has_internal_annotations = True,
    protocol = '2.0.1',
)
