#!/usr/bin/env python

import xbob.db.frgc
import facereclib

database = facereclib.databases.DatabaseXBob(
    database = xbob.db.frgc.Database(),
    name = "frgc",
    image_directory = "/idiap/resource/database/frgc/FRGC-2.0-dist",
    image_extension = ".jpg",
    annotation_directory = "/idiap/user/mguenther/annotations/FRGC",
    annotation_type = 'named',
    protocol = '2.0.1',
)
