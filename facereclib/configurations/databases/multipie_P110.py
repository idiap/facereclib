#!/usr/bin/env python

import bob.db.multipie
import facereclib

multipie_image_directory = "[YOUR_MULTI-PIE_IMAGE_DIRECTORY]"
multipie_annotation_directory = "[YOUR_MULTI-PIE_ANNOTATION_DIRECTORY]"

# here, we only want to have the cameras that are used in the P110 protocol
cameras = ('05_1', '11_0')

database = facereclib.databases.DatabaseBobZT(
    database = bob.db.multipie.Database(
        original_directory = multipie_image_directory,
        annotation_directory = multipie_annotation_directory
    ),
    name = "multipie",
    protocol = 'P110',

    all_files_options = {'cameras' : cameras},
    extractor_training_options = {'cameras' : cameras},
    projector_training_options = {'cameras' : cameras, 'world_sampling': 3, 'world_first': True},
    enroller_training_options = {'cameras' : cameras}
)
