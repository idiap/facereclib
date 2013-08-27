#!/usr/bin/env python

import xbob.db.multipie
import facereclib

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.multipie.Database(),
    name = "multipie",
    original_directory ="/idiap/resource/database/Multi-Pie/data/",
    original_extension = ".png",
    annotation_directory = "/idiap/group/biometric/annotations/multipie/",
    annotation_type = 'multipie',
    protocol = 'U',
)
