#!/usr/bin/env python

import bob.db.xm2vts
import facereclib

xm2vts_directory = "[YOUR_XM2VTS_IMAGE_DIRECTORY]"

# setup for XM2VTS
database = facereclib.databases.DatabaseBob(
    database = bob.db.xm2vts.Database(
        original_directory = xm2vts_directory
    ),
    name = "xm2vts",
    protocol = 'lp1'
)
