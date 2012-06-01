#!/usr/bin/env python

import os
import bob

# The database to use
name = 'LFW'
db = bob.db.lfw.Database()
protocol = 'view1'

img_input_dir = '/idiap/resource/database/lfw/all_images'
img_input_ext = '.jpg'

world_extractor_options = { 'subworld' : 'twofolds' }
world_projector_options = {'subworld' : 'twofolds' } 
world_enroler_options = { 'subworld' : 'twofolds' }

