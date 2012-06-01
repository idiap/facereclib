#!/usr/bin/env python

import bob

# setup for SCface database
name = 'scface'
db = bob.db.scface.Database()
protocol = 'combined'

img_input_dir = "/idiap/group/biometric/databases/scface/images/"
img_input_ext = ".jpg"
pos_input_dir = "/idiap/group/biometric/databases/scface/groundtruths/"
pos_input_ext = ".pos"

first_annot = 0
world_projector_options = { 'subworld': "twothirds" } 

