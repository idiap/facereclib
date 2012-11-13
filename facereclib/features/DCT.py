#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""Features for face recognition"""

import numpy
import math
import bob
from .. import utils

from .Extractor import Extractor

class DCTBlocks (Extractor):

  """Extracts DCT blocks"""
  def __init__(
      self,
      block_size = 12,    # 1 or two parameters for block size
      block_overlap = 11, # 1 or two parameters for block overlap
      number_of_dct_coefficients = 45
  ):

    # call base class constructor
    Extractor.__init__(self)

    # block parameters
    self.m_block_size = block_size if isinstance(block_size, (tuple, list)) else (block_size, block_size)
    self.m_block_overlap = block_overlap if isinstance(block_overlap, (tuple, list)) else (block_overlap, block_overlap)
    self.m_number_of_dct_coefficients = number_of_dct_coefficients
    if self.m_block_size[0] < self.m_block_overlap[0] or self.m_block_size[1] < self.m_block_overlap[1]:
      raise ValueError("The overlap '%s' is bigger than the block size '%s'. This won't work. Please check your setup!"%(self.m_block_overlap, self.m_block_size))
    if self.m_block_size[0] * self.m_block_size[1] <= self.m_number_of_dct_coefficients:
      raise ValueError("You selected more coefficients %d than your blocks have %d. This won't work. Please check your setup!"%(self.m_number_of_dct_coefficients, self.m_block_size[0] * self.m_block_size[1]))


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
    block_shape = bob.ip.get_block_3d_output_shape(prep, self.m_block_size[0], self.m_block_size[1], self.m_block_overlap[0], self.m_block_overlap[1])
    blocks = numpy.ndarray(block_shape, 'float64')
    bob.ip.block(prep, blocks, self.m_block_size[0], self.m_block_size[1], self.m_block_overlap[0], self.m_block_overlap[1])

    if norm_before:
      self._normalize_blocks(blocks)

    if add_xy:
      real_DCT_coef = self.m_number_of_dct_coefficients - 2
    else:
      real_DCT_coef = self.m_number_of_dct_coefficients

    # Initializes cropper and destination array
    DCTF = bob.ip.DCTFeatures(self.m_block_size[0], self.m_block_size[1], self.m_block_overlap[0], self.m_block_overlap[1], real_DCT_coef)

    # Calls the preprocessing algorithm
    dct_blocks = DCTF(blocks)

    n_blocks = block_shape[0]

    dct_blocks_min = 0
    dct_blocks_max = self.m_number_of_dct_coefficients
    TMP_tensor_min = 0
    TMP_tensor_max = self.m_number_of_dct_coefficients

    if norm_before:
      dct_blocks_min += 1
      TMP_tensor_max -= 1

    if add_xy:
      dct_blocks_max -= 2
      TMP_tensor_min += 2

    TMP_tensor = numpy.ndarray((n_blocks, TMP_tensor_max), 'float64')

    #nBlocks = bob.ip.get_n_blocks(prep, self.m_config.BLOCK_H, self.m_config.BLOCK_W, self.m_config.OVERLAP_H, self.m_config.OVERLAP_W)
    # Note: nBlocks = ( n_blocks_h, n_blocks_w, block_h, block_w)
    nBlocks = bob.ip.get_block_4d_output_shape(prep, self.m_block_size[0], self.m_block_size[1], self.m_block_overlap[0], self.m_block_overlap[1])
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




class DCTBlocksVideo(DCTBlocks):

  def __init__(self, **kwargs):
    # call base class constructor with its required parameters
    DCTBlocks.__init__(self, **kwargs)


  def read_feature(self, filename):
    """Read video.FrameContainer containing features extracted from each frame"""
    return utils.video.FrameContainer(str(filename))


  def __call__(self, frame_container):
    """Returns local DCT features computed from each frame in the input video.FrameContainer"""

    output_frame_container = utils.video.FrameContainer()
    for (frame_id, image, quality) in frame_container.frames():
      frame_dcts = DCTBlocks._dct_features(self,image)
      output_frame_container.add_frame(frame_id,frame_dcts,quality)

    return output_frame_container

