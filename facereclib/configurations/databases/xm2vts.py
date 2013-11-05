#!/usr/bin/env python

import xbob.db.xm2vts
import facereclib

xm2vts_directory = "[YOUR_XM2VTS_IMAGE_DIRECTORY]"

# setup for XM2VTS
database = facereclib.databases.DatabaseXBob(
    database = xbob.db.xm2vts.Database(),
    name = "xm2vts",
    original_directory = xm2vts_directory,
    original_extension = ".ppm",
    has_internal_annotations = True,
    protocol = 'lp1'
)
