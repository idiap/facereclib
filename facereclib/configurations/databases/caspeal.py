#!/usr/bin/env python

import bob.db.caspeal
import facereclib

caspeal_directory = "[YOUR_CAS-PEAL_DIRECTORY]"

database = facereclib.databases.DatabaseXBob(
    database = bob.db.caspeal.Database(
      original_directory = caspeal_directory
    ),
    name = "caspeal",
    original_directory = caspeal_directory,
    original_extension = ".tif",
    has_internal_annotations = True,
    protocol = 'lighting'
)
