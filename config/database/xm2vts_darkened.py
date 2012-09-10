#!/usr/bin/env python

import xbob.db.xm2vts

# setup for XM2VTS
name = 'xm2vts'
db = xbob.db.xm2vts.Database()
protocol = 'darkened-lp1'

image_directory = "/idiap/resource/database/xm2vtsdb/images/"
image_extension = ".ppm"
annotation_directory = "/idiap/user/mguenther/annotations/XM2VTS/"
annotation_type = 'eyecenter'

