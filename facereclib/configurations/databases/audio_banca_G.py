#!/usr/bin/env python

import xbob.db.banca
import facereclib
import xbob.db.faceverif_fl

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.faceverif_fl.Database('/idiap/user/ekhoury/databases_protocols/banca/G'),
    name = "audio_banca_p",
    image_directory = "/idiap/temp/ekhoury/databases/banca/wav_from_johnny/",
    image_extension = ".wav",
    protocol = 'G'
)
