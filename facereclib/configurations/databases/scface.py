#!/usr/bin/env python

import bob.db.scface
import facereclib

scface_directory = "[YOUR_SC_FACE_DIRECTORY]"

# setup for SCface database
database = facereclib.databases.DatabaseBobZT(
    database = bob.db.scface.Database(
        original_directory = scface_directory
    ),
    name = 'scface',
    protocol = 'combined'
)
