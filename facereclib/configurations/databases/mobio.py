#!/usr/bin/env python

import xbob.db.mobio
import facereclib

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.mobio.Database(),
    name = "mobio",
    original_directory = "/idiap/resource/database/mobio/IMAGES_PNG",
    original_extension = ".png",
    annotation_directory = "/idiap/resource/database/mobio/IMAGE_ANNOTATIONS",
    annotation_type = 'eyecenter',
    protocol = 'male'
)
