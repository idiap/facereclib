#!/usr/bin/env python

import xbob.db.frgc
import facereclib

frgc_directory = "/idiap/resource/database/frgc/FRGC-2.0-dist"

database = facereclib.databases.DatabaseXBob(
    database = xbob.db.frgc.Database(frgc_directory),
    name = "frgc",
    original_directory = frgc_directory,
    original_extension = ".jpg",
    has_internal_annotations = True,
    protocol = '2.0.1',
)
