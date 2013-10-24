#!/usr/bin/env python

import xbob.db.atnt
import facereclib

atnt_directory = "[YOUR_ATNT_DIRECTORY]"

database = facereclib.databases.DatabaseXBob(
    database = xbob.db.atnt.Database(),
    name = 'atnt',
    original_directory = atnt_directory,
    original_extension = ".pgm"
)
