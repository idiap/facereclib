#!/usr/bin/env python

import xbob.db.lfw

# The database to use
name = 'lfw'
db = xbob.db.lfw.Database()
protocol = 'view1'

image_directory = '/idiap/resource/database/lfw/all_images'
image_extension = '.jpg'

all_files_options = { 'type' : 'unrestricted' }
extractor_training_options = { 'subworld' : 'twofolds', 'type' : 'unrestricted' }
projector_training_options = {'subworld' : 'twofolds', 'type' : 'unrestricted' }
enroller_training_options = { 'subworld' : 'twofolds', 'type' : 'unrestricted' }
features_by_clients_options = { 'subworld' : 'twofolds' }
