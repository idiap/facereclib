#!/usr/bin/env python

import xbob.db.banca
import facereclib

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.banca.Database(),
    name = "banca",
    image_directory = "/idiap/user/mguenther/banca/colored/",
    image_extension = ".ppm",
    annotation_directory = "/idiap/group/vision/visidiap/databases/groundtruth/banca/english/eyecenter/",
    annotation_type = 'eyecenter',
    protocol = 'P'
)
