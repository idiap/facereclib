#!/usr/bin/env python

import xbob.db.mobio
import facereclib

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.mobio.Database(),
    name = "mobio",
    image_directory = "/idiap/temp/ekhoury/databases/MOBIO/denoisedDATA_16k/",
    image_extension = ".sph",
    annotation_directory = "",
    annotation_type = '',
    protocol = 'female',

    all_files_options = { 'gender' : 'female' },
    extractor_training_options = { 'gender' : 'female' },
    projector_training_options = { 'gender' : 'female' },
    enroller_training_options = { 'gender' : 'female' }
)
