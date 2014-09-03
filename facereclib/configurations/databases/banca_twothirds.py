#!/usr/bin/env python

import bob.db.banca
import facereclib

banca_directory = "[YOUR_BANCA_DIRECTORY]"

database = facereclib.databases.DatabaseXBobZT(
    database = bob.db.banca.Database(
        original_directory = banca_directory,
        original_extension = ".ppm"
    ),
    name = "banca",
    original_directory = banca_directory,
    original_extension = ".ppm",
    has_internal_annotations = True,
    protocol = 'P',
    projector_training_options = { 'subworld': "twothirds" }
)
