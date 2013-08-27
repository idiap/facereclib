#!/usr/bin/env python

import xbob.db.arface
import facereclib

database = facereclib.databases.DatabaseXBob(
    database = xbob.db.arface.Database(),
    name = 'arface',
    original_directory = "/idiap/resource/database/AR_Face/images",
    original_extension = ".ppm",
    has_internal_annotations = True,
    protocol = 'all'
)
