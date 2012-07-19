#!/usr/bin/env python

import bob

# setup for XM2VTS
name = 'xm2vts'
db = bob.db.xm2vts.Database()
protocol = 'lp1'

img_input_dir = "/idiap/resource/database/xm2vtsdb/images/frontal/" 
img_input_ext = ".ppm"
pos_input_dir = "/idiap/user/mguenther/annotations/XM2VTS/"
pos_input_ext = ".pos"

first_annot = 0

