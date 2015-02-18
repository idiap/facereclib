#!/usr/bin/env python

import bob.db.atnt
import facereclib

atnt_directory = "/idiap/group/biometric/databases/orl"

database = facereclib.databases.DatabaseBob(
    database = bob.db.atnt.Database(
        original_directory = atnt_directory
    ),
    name = 'atnt'
)
