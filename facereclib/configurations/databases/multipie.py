#!/usr/bin/env python

import xbob.db.multipie
import facereclib

multipie_image_directory = "[YOUR_MULTI-PIE_IMAGE_DIRECTORY]"
multipie_annotation_directory = "[YOUR_MULTI-PIE_ANNOTATION_DIRECTORY]"

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.multipie.Database(),
    name = "multipie",
    original_directory = multipie_image_directory,
    original_extension = ".png",
    annotation_directory = multipie_annotation_directory,
    annotation_type = 'multipie',
    protocol = 'U',
)
