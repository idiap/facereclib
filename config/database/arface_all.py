#!/usr/bin/env python

import xbob.db.arface

# 0/ The database to use
name = 'arface'
db = xbob.db.arface.Database()
protocol = 'all'

image_directory = "/idiap/resource/database/AR_Face/images"
image_extension = ".ppm"
annotation_directory = "/idiap/user/mguenther/annotations/ARface"
annotation_type = 'eyecenter'

