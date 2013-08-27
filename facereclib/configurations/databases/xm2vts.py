#!/usr/bin/env python

import xbob.db.xm2vts
import facereclib

# setup for XM2VTS
database = facereclib.databases.DatabaseXBob(
    database = xbob.db.xm2vts.Database(),
    name = "xm2vts",
    original_directory = "/idiap/resource/database/xm2vtsdb/images/",
    original_extension = ".ppm",
    has_internal_annotations = True,
    protocol = 'lp1'
)
