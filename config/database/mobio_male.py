#!/usr/bin/env python

import xbob.db.mobio

# setup for MoBio database
name = 'mobio'
db = xbob.db.mobio.Database()
protocol = 'male'

image_directory = "/idiap/group/biometric/databases/mobio/still/images/selected-images/"
image_extension = ".jpg"
annotation_directory = "/idiap/group/biometric/annotations/mobio/"
annotation_type = 'eyecenter'

