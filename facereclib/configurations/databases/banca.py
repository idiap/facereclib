#!/usr/bin/env python

import xbob.db.banca
import facereclib

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.banca.Database(),
    name = "banca",
    original_directory = "/idiap/group/biometric/databases/banca/english/images/images/",
    original_extension = ".ppm",
    has_internal_annotations = True,
    protocol = 'P'
)
