#!/usr/bin/env python

import xbob.db.scface
import facereclib

# setup for SCface database
database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.scface.Database(),
    name = 'scface',
    image_directory = "/idiap/group/biometric/databases/scface/images/",
    image_extension = ".jpg",
    annotation_directory = "/idiap/group/biometric/databases/scface/groundtruths/",
    annotation_type = 'scface',
    protocol = 'combined',
    projector_training_options = { 'subworld': "twothirds" }
)
