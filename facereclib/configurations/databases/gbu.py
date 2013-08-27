#!/usr/bin/env python

import xbob.db.gbu
import facereclib

database = facereclib.databases.DatabaseXBob(
    database = xbob.db.gbu.Database(),
    name = "gbu",
    original_directory = "/idiap/resource/database/MBGC-V1",
    original_extension = ".jpg",
    has_internal_annotations = True,
    protocol = 'Good',

    all_files_options = { 'subworld': 'x2' },
    extractor_training_options = { 'subworld': 'x2' },
    projector_training_options = { 'subworld': 'x2' },
    enroller_training_options = { 'subworld': 'x2' }
)
