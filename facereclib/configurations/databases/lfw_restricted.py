#!/usr/bin/env python

import bob.db.lfw
import facereclib

lfw_directory = "[YOUR_LFW_DIRECTORY]"

database = facereclib.databases.DatabaseBob(
    database = bob.db.lfw.Database(
        original_directory = lfw_directory
    ),
    name = 'lfw',
    protocol = 'view1',

    all_files_options = {'world_type' : 'restricted'},
    extractor_training_options = {'world_type' : 'restricted'}, # 'subworld' : 'twofolds'
    projector_training_options = {'world_type' : 'restricted'}, # 'subworld' : 'twofolds'
    enroller_training_options =  {'world_type' : 'restricted'} # 'subworld' : 'twofolds'
)

