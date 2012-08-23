#!/usr/bin/env python

import bob

# setup for Multi-PIE database
name = 'multipie'
db = bob.db.multipie.Database()
protocol = 'P'

img_input_dir = "/idiap/resource/database/Multi-Pie/data/"
img_input_ext = ".png"
pos_input_dir = "/idiap/user/mguenther/annotations/multipie/"
pos_input_ext = ".pos"

annotation_type = 'multipie'
all_files_options = {}
world_extractor_options = {}
world_projector_options = {}
world_enroler_options = {}


keywords = {
  'neutral' : { 'world_cameras' : ['05_1'] },

  'P240' : { 'world_cameras' : ['24_0'] }, # right profile
  'P010' : { 'world_cameras' : ['01_0'] }, #
  'P200' : { 'world_cameras' : ['20_0'] }, # right half-profile
  'P190' : { 'world_cameras' : ['19_0'] }, #

  'P041' : { 'world_cameras' : ['04_1'] }, # right quarter-profile
  'P050' : { 'world_cameras' : ['05_0'] }, #
  'P051' : { 'world_cameras' : ['05_1'] }, # central
  'P140' : { 'world_cameras' : ['14_0'] }, #
  'P130' : { 'world_cameras' : ['13_0'] }, # left quarter-profile


  'P080' : { 'world_cameras' : ['08_0'] }, #
  'P090' : { 'world_cameras' : ['09_0'] }, # left half-profile
  'P120' : { 'world_cameras' : ['12_0'] }, #
  'P110' : { 'world_cameras' : ['11_0'] }, # left profile

  'P191' : { 'world_cameras' : ['19_1'] }, # right quarter-profile from above
  'P081' : { 'world_cameras' : ['08_1'] }  # left quarter-profile from above
}
