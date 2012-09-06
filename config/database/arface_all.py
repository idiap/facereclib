#!/usr/bin/env python

import xbob.db.arface

# 0/ The database to use
name = 'arface'
db = xbob.db.arface.Database()
protocol = 'all'

img_input_dir = "/idiap/resource/database/AR_Face/images"
img_input_ext = ".ppm"
pos_input_dir = "/idiap/user/mguenther/annotations/ARface"
pos_input_ext = ".pos"

annotation_type = 'eyecenter'

