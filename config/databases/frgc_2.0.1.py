#!/usr/bin/env python

import xbob.db.frgc

# setup for SCface database
name = 'frgc'
db = xbob.db.frgc.Database()
protocol = '2.0.1'

image_directory = "/idiap/resource/database/frgc/FRGC-2.0-dist"
image_extension = ".jpg"
annotation_directory = "/idiap/user/mguenther/annotations/FRGC"
annotation_type = 'named'

