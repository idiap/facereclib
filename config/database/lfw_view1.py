#!/usr/bin/env python

import xbob.db.lfw

# The database to use
name = 'lfw'
db = xbob.db.lfw.Database()
protocol = 'view1'

image_directory = '/idiap/resource/database/lfw/all_images'
image_extension = '.jpg'

extractor_training_options = { 'subworld' : 'twofolds' }
projector_training_options = {'subworld' : 'twofolds' }
enroler_training_options = { 'subworld' : 'twofolds' }

