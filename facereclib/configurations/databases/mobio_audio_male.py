#!/usr/bin/env python

import xbob.db.mobio
import facereclib

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.mobio.Database(),
    name = "mobio",
    image_directory = "/idiap/temp/ekhoury/databases/MOBIO/denoisedDATA_16k/",
    image_extension = ".sph",
    protocol = 'male',

    all_files_options = { 'gender' : 'male' },
    extractor_training_options = { 'gender' : 'male' },
    projector_training_options = { 'gender' : 'male' },
    enroller_training_options = { 'gender' : 'male' }
)
