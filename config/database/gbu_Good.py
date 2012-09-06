#!/usr/bin/env python

import xbob.db.gbu

# The database to use
name = 'gbu'
db = xbob.db.gbu.Database()
protocol = 'Good'

img_input_dir = '/idiap/resource/database/MBGC-V1'
img_input_ext = '.jpg'
pos_input_dir = '/idiap/user/mguenther/annotations/GBU'
pos_input_ext = '.pos'
annotation_type = 'named'

all_files_options = { 'subworld': 'x2' }
world_extractor_options = { 'subworld': 'x2' }
world_projector_options = { 'subworld': 'x2' }
world_enroler_options = { 'subworld': 'x2' }
