#!/usr/bin/env python

import xbob.db.multipie
import facereclib

# here, we only want to have the cameras that are used in the P110 protocol
cameras = ('05_1', '11_0')

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.multipie.Database(),
    name = "multipie",
    original_directory ="/idiap/resource/database/Multi-Pie/data/",
    original_extension = ".png",
    annotation_directory = "/idiap/group/biometric/annotations/multipie/",
    annotation_type = 'multipie',
    protocol = 'P110',
    all_files_options = {'cameras' : cameras},
    extractor_training_options = {'cameras' : cameras},
    projector_training_options = {'cameras' : cameras, 'world_sampling': 3, 'world_first': True},
    enroller_training_options = {'cameras' : cameras}
)
