#!/usr/bin/env python

import xbob.db.scface

# setup for SCface database
name = 'scface'
db = xbob.db.scface.Database()
protocol = 'combined'

image_directory = "/idiap/group/biometric/databases/scface/images/"
image_extension = ".jpg"
annotation_directory = "/idiap/group/biometric/databases/scface/groundtruths/"
annotation_type = 'scface'

projector_training_options = { 'subworld': "twothirds" }

