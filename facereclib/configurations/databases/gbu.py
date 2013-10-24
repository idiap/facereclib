#!/usr/bin/env python

import xbob.db.gbu
import facereclib

mbgc_v1_directory = "[YOUR_MBGC-V1_DIRECTORY]"

database = facereclib.databases.DatabaseXBob(
    database = xbob.db.gbu.Database(),
    name = "gbu",
    original_directory = mbgc_v1_directory,
    original_extension = ".jpg",
    has_internal_annotations = True,
    protocol = 'Good',

    all_files_options = { 'subworld': 'x2' },
    extractor_training_options = { 'subworld': 'x2' },
    projector_training_options = { 'subworld': 'x2' },
    enroller_training_options = { 'subworld': 'x2' }
)
