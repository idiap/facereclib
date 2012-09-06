#!/usr/bin/env python

import xbob.db.xm2vts

# setup for XM2VTS
name = 'xm2vts'
db = xbob.db.xm2vts.Database()
protocol = 'darkened-lp1'

img_input_dir = "/idiap/resource/database/xm2vtsdb/images/"
img_input_ext = ".ppm"
pos_input_dir = "/idiap/user/mguenther/annotations/XM2VTS/"
pos_input_ext = ".pos"

annotation_type = 'eyecenter'

