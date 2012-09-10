#!/usr/bin/env python

import xbob.db.multipie

# setup for Multi-PIE database
name = 'multipie'
db = xbob.db.multipie.Database()
protocol = 'P110'

image_directory = "/idiap/resource/database/Multi-Pie/data/"
image_extension = ".png"
annotation_directory = "/idiap/group/biometric/annotations/multipie/"
annotation_type = 'multipie'

projector_training_options = { 'world_sampling': 3, 'world_first': True }

