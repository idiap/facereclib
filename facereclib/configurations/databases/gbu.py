#!/usr/bin/env python

import bob.db.gbu
import facereclib

mbgc_v1_directory = "[YOUR_MBGC-V1_DIRECTORY]"

database = facereclib.databases.DatabaseBob(
    database = bob.db.gbu.Database(
        original_directory = mbgc_v1_directory
    ),
    name = "gbu",
    protocol = 'Good',

    all_files_options = { 'subworld': 'x2' },
    extractor_training_options = { 'subworld': 'x2' },
    projector_training_options = { 'subworld': 'x2' },
    enroller_training_options = { 'subworld': 'x2' }
)
