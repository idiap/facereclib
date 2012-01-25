#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import os 
import bob
import utils
import numpy as np


def compute(img_input, pos_input, prep_output,
  CROP_EYES_D, CROP_H, CROP_W, CROP_OH, CROP_OW,
  GAMMA, SIGMA0, SIGMA1, SIZE, THRESHOLD, ALPHA,
  first_annot, force):

  # Initialize cropper and destination array
  FEN = bob.ip.FaceEyesNorm( CROP_EYES_D, CROP_H, CROP_W, CROP_OH, CROP_OW)
  cropped_img = np.ndarray((CROP_H, CROP_W), 'float64') 

  # Initialize the Tan and Triggs preprocessing
  TT = bob.ip.TanTriggs( GAMMA, SIGMA0, SIGMA1, SIZE, THRESHOLD, ALPHA)
  preprocessed_img = np.ndarray((CROP_H, CROP_W), 'float64') #bob.core.array.float64_2(CROP_H, CROP_W)

  # process the 'dictionary of files'
  for k in img_input:
    if force == True and os.path.exists(prep_output[k]):
      print "Remove old preprocessed image %s." % (prep_output[k])
      os.remove(prep_output[k])

    if os.path.exists(prep_output[k]):
      print "Preprocessed image %s already exists."  % (prep_output[k])
    else:
      print "Preprocessing sample %s with Tan and Triggs." % (img_input[k])

      # input image file
      img_unk = bob.io.load( str(img_input[k]) )
      
      # convert to grayscale
      if(img_unk.ndim == 3):
        img = bob.ip.rgb_to_gray(img_unk)
      else:
        img = img_unk

      # input eyes position file
      annots = [int(j.strip()) for j in open(pos_input[k]).read().split()]
      if first_annot == 0: LW, LH, RW, RH = annots[0:4]
      else: nb_annots, LW, LH, RW, RH = annots[0:5]

      # extract and crop a face 
      FEN(img, cropped_img, LH, LW, RH, RW) 
      # preprocess a face using Tan and Triggs
      TT(cropped_img, preprocessed_img)

      # vectorize and save
      prep_img_1d = np.reshape(preprocessed_img, preprocessed_img.size)
      #prep_img_1d = bob.core.array.convert(preprocessed_img, 'uint8', sourceRange=(-THRESHOLD,THRESHOLD), destRange=(0,255))
      utils.ensure_dir(os.path.dirname(str(prep_output[k])))
      bob.io.save(prep_img_1d,str(prep_output[k]))
