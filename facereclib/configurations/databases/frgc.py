#!/usr/bin/env python

import bob.db.frgc
import facereclib

frgc_directory = "[YOUR_FRGC_DIRECTORY]"

database = facereclib.databases.DatabaseBob(
    database = bob.db.frgc.Database(frgc_directory),
    name = "frgc",
    protocol = '2.0.1',
)
