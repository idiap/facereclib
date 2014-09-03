#!/usr/bin/env python

import bob.db.arface
import facereclib

arface_directory = "[YOUR_ARFACE_DIRECTORY]"

database = facereclib.databases.DatabaseBob(
    database = bob.db.arface.Database(
        original_directory = arface_directory
    ),
    name = 'arface',
    protocol = 'all'
)
