#!/usr/bin/env python

import xbob.db.mobio

# setup for MoBio database
name = 'mobio'
db = xbob.db.mobio.Database()
protocol = 'female'

img_input_dir = "/idiap/group/biometric/databases/mobio/still/images/selected-images/"
img_input_ext = ".jpg"
pos_input_dir = "/idiap/group/biometric/annotations/mobio/"
pos_input_ext = ".pos"

annotation_type = 'eyecenter'

