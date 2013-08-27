#!/usr/bin/env python

import xbob.db.lfw
import facereclib

database = facereclib.databases.DatabaseXBob(
    database = xbob.db.lfw.Database(),
    name = 'lfw',
    original_directory = "/idiap/resource/database/lfw/all_images",
    original_extension = ".jpg",
    protocol = 'view1',

    all_files_options = {'world_type' : 'unrestricted'},
    extractor_training_options = {'world_type' : 'unrestricted'}, # 'subworld' : 'twofolds'
    projector_training_options = {'world_type' : 'unrestricted'}, # 'subworld' : 'twofolds'
    enroller_training_options =  {'world_type' : 'unrestricted'} # 'subworld' : 'twofolds'
)
