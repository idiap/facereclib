#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""Features for face recognition"""

import numpy,math
import bob

class DCTBlocks:
  """Extracts DCT blocks"""
  def __init__(self, config):
    self.m_config = config

  def _normalize_blocks(self, src):
    for i in range(src.shape[0]):
      block = src[i, :, :]
      mean = numpy.mean(block)
      std = numpy.sum((block - mean) ** 2) / block.size
      if std == 0:
        std = 1
      else:
        std = math.sqrt(std)

      src[i, :, :] = (block - mean) / std

        
  def _normalize_dct(self, src):
    for i in range(src.shape[1]):
      col = src[:, i]
      mean = numpy.mean(col)
      std = numpy.sum((col - mean) ** 2) / col.size
      if std == 0:
        std = 1
      else:
        std = math.sqrt(std)

      src[:, i] = (col - mean) / std


  def _dct_features(self, prep, norm_before = True, norm_after = True, add_xy = False):
    # REVERT TO bob-1.0.4 block_shape = bob.ip.get_block_3d_output_shape(prep, self.m_config.BLOCK_H, self.m_config.BLOCK_W, self.m_config.OVERLAP_H, self.m_config.OVERLAP_W)
    block_shape = bob.ip.get_block_shape(prep, self.m_config.BLOCK_H, self.m_config.BLOCK_W, self.m_config.OVERLAP_H, self.m_config.OVERLAP_W)
    blocks = numpy.ndarray(block_shape, 'float64')
    bob.ip.block(prep, blocks, self.m_config.BLOCK_H, self.m_config.BLOCK_W, self.m_config.OVERLAP_H, self.m_config.OVERLAP_W)

    if norm_before:
      self._normalize_blocks(blocks)

    if add_xy:
      real_DCT_coef = self.m_config.N_DCT_COEF - 2
    else:
      real_DCT_coef = self.m_config.N_DCT_COEF

    
    # Initializes cropper and destination array
    DCTF = bob.ip.DCTFeatures(self.m_config.BLOCK_H, self.m_config.BLOCK_W, self.m_config.OVERLAP_H, self.m_config.OVERLAP_W, real_DCT_coef)
    
    # Calls the preprocessing algorithm
    dct_blocks = DCTF(blocks)

    n_blocks = block_shape[0]

    dct_blocks_min = 0
    dct_blocks_max = self.m_config.N_DCT_COEF
    TMP_tensor_min = 0
    TMP_tensor_max = self.m_config.N_DCT_COEF

    if norm_before:
      dct_blocks_min += 1
      TMP_tensor_max -= 1

    if add_xy:
      dct_blocks_max -= 2
      TMP_tensor_min += 2
    
    TMP_tensor = numpy.ndarray((n_blocks, TMP_tensor_max), 'float64')
    
    # REVERT TO bob-1.0.4
    #nBlocks = bob.ip.get_n_blocks(prep, self.m_config.BLOCK_H, self.m_config.BLOCK_W, self.m_config.OVERLAP_H, self.m_config.OVERLAP_W)
    ## Note: nBlocks = ( n_blocks_h, n_blocks_w, block_h, block_w)
    #nBlocks = bob.ip.get_block_4d_output_shape(prep, self.m_config.BLOCK_H, self.m_config.BLOCK_W, self.m_config.OVERLAP_H, self.m_config.OVERLAP_W)
    nBlocks = bob.ip.get_n_blocks(prep, self.m_config.BLOCK_H, self.m_config.BLOCK_W, self.m_config.OVERLAP_H, self.m_config.OVERLAP_W)
    for by in range(nBlocks[0]):
      for bx in range(nBlocks[1]):
        bi = bx + by * nBlocks[1]
        if add_xy:
          TMP_tensor[bi, 0] = bx
          TMP_tensor[bi, 1] = by
        
        TMP_tensor[bi, TMP_tensor_min:TMP_tensor_max] = dct_blocks[bi, dct_blocks_min:dct_blocks_max]

    if norm_after:
      self._normalize_dct(TMP_tensor)

    return TMP_tensor


  def __call__(self, image):
    """Computes and returns DCT blocks for the given input image"""

    # Computes DCT features
    return self._dct_features(image)

