#!/usr/bin/env python

import os
import bob

# The database to use
name = 'GBU'
db = bob.db.gbu.Database()
protocol = 'Good'

img_input_dir = '/idiap/resource/database/MBGC-V1'
img_input_ext = '.jpg'

all_files_options = { 'subworld': 'x2' }
world_extractor_options = { 'subworld': 'x2' }
world_projector_options = { 'subworld': 'x2' } 
world_enroler_options = { 'subworld': 'x2' }
