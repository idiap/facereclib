#!/usr/bin/env python

import xbob.db.banca
import facereclib

banca_directory = "[YOUR_BANCA_DIRECTORY]"

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.banca.Database(),
    name = "banca",
    original_directory = banca_directory,
    original_extension = ".ppm",
    has_internal_annotations = True,
    protocol = 'P'
)
