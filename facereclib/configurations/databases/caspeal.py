#!/usr/bin/env python

import bob.db.caspeal
import facereclib

caspeal_directory = "[YOUR_CAS-PEAL_DIRECTORY]"

database = facereclib.databases.DatabaseBob(
    database = bob.db.caspeal.Database(
        original_directory = caspeal_directory
    ),
    name = "caspeal",
    protocol = 'lighting'
)
