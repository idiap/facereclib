#!/usr/bin/env python

import bob.db.atnt
import facereclib

atnt_directory = "[YOUR_ATNT_DIRECTORY]"

database = facereclib.databases.DatabaseXBob(
    database = bob.db.atnt.Database(
      original_directory = atnt_directory
    ),
    name = 'atnt',
    original_directory = atnt_directory,
    original_extension = ".pgm"
)
