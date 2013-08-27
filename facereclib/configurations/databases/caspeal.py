#!/usr/bin/env python

import xbob.db.caspeal
import facereclib

database = facereclib.databases.DatabaseXBob(
    database = xbob.db.caspeal.Database(),
    name = "caspeal",
    original_directory = "/idiap/resource/database/CAS-PEAL",
    original_extension = ".tif",
    has_internal_annotations = True,
    protocol = 'lighting'
)
