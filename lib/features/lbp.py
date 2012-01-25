#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import os, math
import bob
import utils
import numpy as np

def compute(img_input, pos_input, features_output,
  CROP_EYES_D, CROP_H, CROP_W, CROP_OH, CROP_OW,
  GAMMA, SIGMA0, SIGMA1, SIZE, THRESHOLD, ALPHA,
  BLOCK_H, BLOCK_W, OVERLAP_H, OVERLAP_W, RADIUS, P_N, CIRCULAR, 
  TO_AVERAGE, ADD_AVERAGE_BIT, UNIFORM, ROT_INV,
  first_annot, force):

  # Initializes cropper and destination array
  FEN = bob.ip.FaceEyesNorm( CROP_EYES_D, CROP_H, CROP_W, CROP_OH, CROP_OW)
  cropped_img = np.ndarray((CROP_H, CROP_W), 'float64')

  # Initializes the Tan and Triggs preprocessing
  TT = bob.ip.TanTriggs( GAMMA, SIGMA0, SIGMA1, SIZE, THRESHOLD, ALPHA)
  preprocessed_img = np.ndarray((CROP_H, CROP_W), 'float64')

  # Initializes LBPHS processor
  LBPHSF = bob.ip.LBPHSFeatures( BLOCK_H, BLOCK_W, OVERLAP_H, OVERLAP_W, RADIUS, P_N, CIRCULAR, TO_AVERAGE, ADD_AVERAGE_BIT, UNIFORM, ROT_INV)

  # Processes the 'dictionary of files'
  for k in img_input:
    print img_input[k]
    if force == True and os.path.exists(features_output[k]):
      print "Remove old features %s." % (features_output[k])
      os.remove(features_output[k])

    exist = True
    if not os.path.exists(features_output[k]):
        exist = False

    if exist:
      print "Features for sample %s already exists."  % (img_input[k])
    else:
      print "Computing features from sample %s." % (img_input[k])

      # Loads image file
      img_unk = bob.io.load( str(img_input[k]) )
      
      # Converts to grayscale
      if(img_unk.ndim == 3):
        img = bob.ip.rgb_to_gray(img_unk)
      else:
        img = img_unk

      # Input eyes position file
      annots = [int(j.strip()) for j in open(pos_input[k]).read().split()]
      if first_annot == 0: LW, LH, RW, RH = annots[0:4]
      else: nb_annots, LW, LH, RW, RH = annots[0:5]

      # Extracts and crops a face 
      FEN(img, cropped_img, LH, LW, RH, RW) 
       
      # Preprocesses a face using Tan and Triggs
      TT(cropped_img, preprocessed_img)
      preprocessed_img_s=preprocessed_img.convert('uint8', sourceRange=(-THRESHOLD,THRESHOLD))

      # Computes LBP histograms
      lbphs_blocks = LBPHSF(preprocessed_img_s)
      # Converts to Blitz++ array
      lbphs_array = np.ndarray((len(lbphs_blocks) * LBPHSF.NBins,), 'float64')
      for bi in range(0,len(lbphs_blocks)):
        for j in range(LBPHSF.NBins):
          lbphs_array[bi*LBPHSF.NBins+j] = lbphs_blocks[bi][j]
      # Saves to file
      utils.ensure_dir(os.path.dirname(str(features_output[k])))
      bob.io.save(lbphs_array,str(features_output[k]))
