#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy

class LGBPHS:
  """Extractor for local Gabor binary pattern histogram sequences"""
  
  def __init__(self, setup):
    """Initializes the local Gabor binary pattern histogram sequence tool chain with the given file selector object"""
    # Initializes LBPHS processor
    real_h = setup.BLOCK_H + 2 * setup.RADIUS
    real_w = setup.BLOCK_W + 2 * setup.RADIUS
    real_oh = setup.OVERLAP_H + 2 * setup.RADIUS
    real_ow = setup.OVERLAP_W + 2 * setup.RADIUS
    self.m_lgbphs_extractor = bob.ip.LBPHSFeatures(
          real_h, real_w, real_oh, real_ow, 
          setup.RADIUS, setup.P_N, setup.CIRCULAR,
          setup.TO_AVERAGE, setup.ADD_AVERAGE_BIT, setup.UNIFORM, setup.ROT_INV
    )
    
    self.m_gwt = bob.ip.GaborWaveletTransform(
          number_of_angles = setup.GABOR_DIRECTIONS,
          number_of_scales = setup.GABOR_SCALES,
          sigma = setup.GABOR_SIGMA, 
          k_max = setup.GABOR_K_MAX,
          k_fac = setup.GABOR_K_FAC, 
          pow_of_k = setup.GABOR_POW_OF_K,
          dc_free = setup.GABOR_DC_FREE
    )
    self.m_trafo_image = None
    self.m_split = setup.SPLIT
    self.m_use_phases = setup.USE_PHASES
    self.m_sparse = setup.SPARSE if hasattr(setup, 'SPARSE') else False
    if self.m_sparse and self.m_split:
      raise ValueError("Sparse histograms cannot be split! Check your setup!")
  
  
  def __fill__(self, lgbphs_array, lgbphs_blocks, j):
    """Copies the given array into the given blocks""" 
    # fill array in the desired shape
    if self.m_split == None:
      start = j * self.m_n_bins * self.m_n_blocks
      for b in range(self.m_n_blocks):
        lgbphs_array[start + b * self.m_n_bins : start + (b+1) * self.m_n_bins] = lgbphs_blocks[b][:]
    elif self.m_split == 'blocks':
      for b in range(self.m_n_blocks):
        lgbphs_array[b, j * self.m_n_bins : (j+1) * self.m_n_bins] = lgbphs_blocks[b][:]
    elif self.m_split == 'wavelets':
      for b in range(self.m_n_blocks):
        lgbphs_array[j, b * self.m_n_bins : (b+1) * self.m_n_bins] = lgbphs_blocks[b][:]
    elif self.m_split == 'both':
      for b in range(self.m_n_blocks):
        lgbphs_array[j * self.m_n_blocks + b, 0 : self.m_n_bins] = lgbphs_blocks[b][:]
    
  def __sparsify__(self, array):
    assert len(array.shape) == 1 
    indices = []
    values = []
    for i in range(array.shape[0]):
      if array[i] != 0.:
        indices.append(i)
        values.append(array[i])

    return numpy.array([indices, values], dtype = numpy.float64)
  
  def __call__(self, image):
    """Extracts the local Gabor binary pattern histogram sequence from the given image"""
    # perform GWT on image
    if self.m_trafo_image == None or self.m_trafo_image.shape[1:2] != image.shape:
      # create trafo image
      self.m_trafo_image = self.m_gwt.empty_trafo_image(image)
      
    # convert image to complex
    image = image.astype(numpy.complex128)
    self.m_gwt(image, self.m_trafo_image)
    
    jet_length = self.m_gwt.number_of_kernels * (2 if self.m_use_phases else 1) 
    
    lgbphs_array = None
    # iterate through the layers of the trafo image
    for j in range(self.m_gwt.number_of_kernels):
      # compute absolute part of complex response
      abs_image = numpy.abs(self.m_trafo_image[j])
      # Computes LBP histograms
      abs_blocks = self.m_lgbphs_extractor(abs_image)
      
      # Converts to Blitz array (of different dimensionalities)
      self.m_n_bins = self.m_lgbphs_extractor.n_bins
      self.m_n_blocks = len(abs_blocks)

      if self.m_split == None:
        shape = (self.m_n_blocks * self.m_n_bins * jet_length,)  
      elif self.m_split == 'blocks':
        shape = (self.m_n_blocks, self.m_n_bins * jet_length)
      elif self.m_split == 'wavelets':
        shape = (jet_length, self.m_n_bins * self.m_n_blocks)
      elif self.m_split == 'both':
        shape = (jet_length * self.m_n_blocks, self.m_n_bins)
      else:
        raise ValueError("The split parameter must be one of ['blocks', 'wavelets', 'both'] or None")
     
      # create new array if not done yet
      if lgbphs_array == None:
        lgbphs_array = numpy.ndarray(shape, 'float64')

      # fill the array with the absolute values of the Gabor wavelet transform
      self.__fill__(lgbphs_array, abs_blocks, j)
      
      if self.m_use_phases:
        # compute phase part of complex response
        phase_image = numpy.angle(self.m_trafo_image[j])
        # Computes LBP histograms
        phase_blocks = self.m_lgbphs_extractor(phase_image)
        # fill the array with the phases at the end of the blocks
        self.__fill__(lgbphs_array, phase_blocks, j + self.m_gwt.number_of_kernels)
        

    # return the concatenated list of all histograms
    return self.__sparsify__(lgbphs_array) if self.m_sparse else lgbphs_array

