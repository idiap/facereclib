#!/usr/bin/env python

import bob.db.multipie
import facereclib

multipie_image_directory = "[YOUR_MULTI-PIE_IMAGE_DIRECTORY]"
multipie_annotation_directory = "[YOUR_MULTI-PIE_ANNOTATION_DIRECTORY]"

database = facereclib.databases.DatabaseBobZT(
    database = bob.db.multipie.Database(
        original_directory = multipie_image_directory,
        annotation_directory = multipie_annotation_directory
    ),
    name = "multipie",
    protocol = 'U',
)
