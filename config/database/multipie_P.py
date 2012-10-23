#!/usr/bin/env python

import xbob.db.multipie
import facereclib

# here, we only want to have the cameras that are used in the P protocol
cameras = ('24_0', '01_0', '20_0', '19_0', '04_1', '05_0', '05_1', '14_0', '13_0', '08_0', '09_0', '12_0', '11_0')

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.multipie.Database(),
    name = "multipie",
    image_directory ="/idiap/resource/database/Multi-Pie/data/",
    image_extension = ".png",
    annotation_directory = "/idiap/group/biometric/annotations/multipie/",
    annotation_type = 'multipie',
    protocol = 'P',
    all_files_options = {'cameras' : cameras},
    extractor_training_options = {'cameras' : cameras},
    projector_training_options = {'cameras' : cameras, 'world_sampling': 3, 'world_first': True},
    enroller_training_options = {'cameras' : cameras}
)

# this probably is garbage and will be removed soon

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
