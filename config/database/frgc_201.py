#!/usr/bin/env python

import xbob.db.frgc

# setup for SCface database
name = 'frgc'
db = xbob.db.frgc.Database()
protocol = '2.0.1'

img_input_dir = "/idiap/resource/database/frgc/FRGC-2.0-dist"
img_input_ext = ".jpg"
pos_input_dir = "/idiap/user/mguenther/annotations/FRGC"
pos_input_ext = ".pos"

annotation_type = 'named'

