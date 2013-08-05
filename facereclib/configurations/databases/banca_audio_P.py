#!/usr/bin/env python

import xbob.db.banca
import facereclib
import xbob.db.verification.filelist

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.verification.filelist.Database('/idiap/user/ekhoury/databases_protocols/banca/P1'),
    name = "audio_banca_p",
    image_directory = "/idiap/temp/ekhoury/databases/banca/wav_from_johnny/",
    image_extension = ".wav",
    protocol = 'P'
)
