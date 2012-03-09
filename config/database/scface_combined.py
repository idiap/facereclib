#!/usr/bin/env python

import bob

# setup for SCface database
name = 'scface'
db = bob.db.scface.Database()
protocol = 'combined'

img_input_dir = "/idiap/temp/lelshafey/databases/scface/images/"
img_input_ext = ".jpg"
pos_input_dir = "/idiap/temp/lelshafey/databases/scface/groundtruths/"
pos_input_ext = ".pos"

first_annot = 0
all_files_options = {}
world_extractor_options = {}
world_projector_options = { 'subworld': "twothirds" } 
world_enroler_options = {}

