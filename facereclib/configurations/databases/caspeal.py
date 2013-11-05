#!/usr/bin/env python

import xbob.db.caspeal
import facereclib

caspeal_directory = "[YOUR_CAS-PEAL_DIRECTORY]"

database = facereclib.databases.DatabaseXBob(
    database = xbob.db.caspeal.Database(),
    name = "caspeal",
    original_directory = caspeal_directory,
    original_extension = ".tif",
    has_internal_annotations = True,
    protocol = 'lighting'
)
