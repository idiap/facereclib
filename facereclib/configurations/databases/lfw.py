#!/usr/bin/env python

import xbob.db.lfw
import facereclib

lfw_directory = "[YOUR_LFW_DIRECTORY]"

database = facereclib.databases.DatabaseXBob(
    database = xbob.db.lfw.Database(),
    name = 'lfw',
    original_directory = lfw_directory,
    original_extension = ".jpg",
    protocol = 'view1'
)

