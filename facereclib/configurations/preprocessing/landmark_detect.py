#!/usr/bin/env python

import facereclib

# define the preprocessor
preprocessor = facereclib.preprocessing.FaceDetector(
    post_processor = 'face-crop',
    use_flandmark = True
)
