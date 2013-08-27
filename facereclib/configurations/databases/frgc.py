#!/usr/bin/env python

import xbob.db.frgc
import facereclib

database = facereclib.databases.DatabaseXBob(
    database = xbob.db.frgc.Database(),
    name = "frgc",
    original_directory = "/idiap/resource/database/frgc/FRGC-2.0-dist",
    original_extension = ".jpg",
    has_internal_annotations = True,
    protocol = '2.0.1',
)
