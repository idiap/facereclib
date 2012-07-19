#!/usr/bin/env python

import bob

# setup for MoBio database
name = 'mobio'
db = bob.db.mobio.Database()
protocol = 'male'

img_input_dir = "/idiap/group/biometric/databases/mobio/still/images/selected-images/"
img_input_ext = ".jpg"
pos_input_dir = "/idiap/group/biometric/annotations/mobio/"
pos_input_ext = ".pos"

annotation_type = 'eyecenter'
world_projector_options = { 'subworld': "twothirds" }

