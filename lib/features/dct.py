#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import os, math
import utils
import bob
import numpy as np

def normalizeBlocks(src):
  for i in range(src.shape[0]):
    block = src[i, :, :]
    mean = np.mean(block)
    std = np.sum((block - mean) ** 2) / block.size
    if std == 0:
      std = 1
    else:
      std = math.sqrt(std)

    src[i, :, :] = (block - mean) / std

    
def normalizeDCT(src):
  for i in range(src.shape[1]):
    col = src[:, i]
    mean = np.mean(col)
    std = np.sum((col - mean) ** 2) / col.size
    if std == 0:
      std = 1
    else:
      std = math.sqrt(std)

    src[:, i] = (col - mean) / std


def dctfeatures(prep, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W, 
    A_N_DCT_COEF, norm_before, norm_after, add_xy):
  
  blockShape = bob.ip.getBlockShape(prep, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)
  blocks = np.ndarray(blockShape, 'float64')
  bob.ip.block(prep, blocks, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)

  if norm_before:
    normalizeBlocks(blocks)

  if add_xy:
    real_DCT_coef = A_N_DCT_COEF - 2
  else:
    real_DCT_coef = A_N_DCT_COEF

  
  # Initializes cropper and destination array
  DCTF = bob.ip.DCTFeatures(A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W, real_DCT_coef)
  
  # Calls the preprocessing algorithm
  dct_blocks = DCTF(blocks)

  n_blocks = blockShape[0]

  dct_blocks_min = 0
  dct_blocks_max = A_N_DCT_COEF
  TMP_tensor_min = 0
  TMP_tensor_max = A_N_DCT_COEF

  if norm_before:
    dct_blocks_min += 1
    TMP_tensor_max -= 1

  if add_xy:
    dct_blocks_max -= 2
    TMP_tensor_min += 2
  
  TMP_tensor = np.ndarray((n_blocks, TMP_tensor_max), 'float64')
  
  nBlocks = bob.ip.getNBlocks(prep, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)
  for by in range(nBlocks[0]):
    for bx in range(nBlocks[1]):
      bi = bx + by * nBlocks[1]
      if add_xy:
        TMP_tensor[bi, 0] = bx
        TMP_tensor[bi, 1] = by
      
      TMP_tensor[bi, TMP_tensor_min:TMP_tensor_max] = dct_blocks[bi, dct_blocks_min:dct_blocks_max]

  if norm_after:
    normalizeDCT(TMP_tensor)

  return TMP_tensor


def compute(img_input, pos_input, features_output,
  CROP_EYES_D, CROP_H, CROP_W, CROP_OH, CROP_OW,
  GAMMA, SIGMA0, SIGMA1, SIZE, THRESHOLD, ALPHA,
  BLOCK_H, BLOCK_W, OVERLAP_H, OVERLAP_W, N_DCT_COEF, first_annot, force):

  # Initializes cropper and destination array
  FEN = bob.ip.FaceEyesNorm( CROP_EYES_D, CROP_H, CROP_W, CROP_OH, CROP_OW)
  cropped_img = np.ndarray((CROP_H, CROP_W), 'float64')

  # Initializes the Tan and Triggs preprocessing
  TT = bob.ip.TanTriggs( GAMMA, SIGMA0, SIGMA1, SIZE, THRESHOLD, ALPHA)
  preprocessed_img = np.ndarray((CROP_H, CROP_W), 'float64')

  # Processes the 'dictionary of files'
  for k in img_input:
    if force == True and os.path.exists(features_output[k]):
      print "Remove old features %s." % (features_output[k])
      os.remove(features_output[k])

    if os.path.exists(features_output[k]):
      print "Features %s already exists."  % (features_output[k])
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
      # Computes DCT features
      dct_blocks=dctfeatures(preprocessed_img, BLOCK_H, BLOCK_W, OVERLAP_H, OVERLAP_W, N_DCT_COEF,
        True, True, False)

      # Saves to file
      utils.ensure_dir(os.path.dirname(str(features_output[k])))
      bob.io.save(dct_blocks,str(features_output[k]))
