#!/usr/bin/env python

import bob

# 0/ The database to use
name = 'arface'
db = bob.db.arface.Database()
protocol = 'all'

img_input_dir = "/idiap/resource/database/AR_Face/images"
img_input_ext = ".ppm"
pos_input_dir = "/idiap/user/mguenther/annotations/ARface"
pos_input_ext = ".pos"

first_annot = 0

