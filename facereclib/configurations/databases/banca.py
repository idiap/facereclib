#!/usr/bin/env python

import bob.db.banca
import facereclib

banca_directory = "[YOUR_BANCA_DIRECTORY]"

database = facereclib.databases.DatabaseBobZT(
    database = bob.db.banca.Database(
        original_directory = banca_directory,
        original_extension = '.ppm'
    ),
    name = "banca",
    protocol = 'P'
)
