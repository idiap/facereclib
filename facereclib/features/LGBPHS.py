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
    self.m_lgbphs_extractor = bob.ip.LBPHSFeatures(
          setup.BLOCK_H, setup.BLOCK_W, setup.OVERLAP_H, setup.OVERLAP_W, 
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
  
  def __call__(self, image):
    """Extracts the local Gabor binary pattern histogram sequence from the given image"""
    # perform GWT on image
    if self.m_trafo_image == None or self.m_trafo_image.shape[1:2] != image.shape:
      # create jet image
      self.m_trafo_image = self.m_gwt.empty_trafo_image(image)
      
    # convert image to complex
    image = image.astype(numpy.complex128)
    self.m_gwt(image, self.m_trafo_image)
    
    lgbphs_array = None
    # iterate through the layers of the trafo image
    for j in range(self.m_gwt.number_of_kernels):
      # compute absolute part of complex response
      abs_image = numpy.abs(self.m_trafo_image[j,:,:])
      # Computes LBP histograms
      lbphs_blocks = self.m_lgbphs_extractor(abs_image)
      
      # Converts to Blitz array (of different dimensionalities)
      n_bins = self.m_lgbphs_extractor.n_bins
      n_blocks = len(lbphs_blocks)

      if self.m_split == None:
        shape = (n_blocks * n_bins * self.m_gwt.number_of_kernels,)  
      elif self.m_split == 'blocks':
        shape = (n_blocks, n_bins * self.m_gwt.number_of_kernels)
      elif self.m_split == 'wavelets':
        shape = (self.m_gwt.number_of_kernels, n_bins * n_blocks)
      elif self.m_split == 'both':
        shape = (self.m_gwt.number_of_kernels * n_blocks, n_bins)
      
      # create new array if not done yet
      if lgbphs_array == None:
        lgbphs_array = numpy.ndarray(shape, 'float64')

      # fill array in the desired shape
      if self.m_split == None:
        lgbphs_array[j*n_blocks*n_bins:(j+1)*n_blocks*n_bins] = lbphs_blocks.flatten()
      elif self.m_split == 'blocks':
        for b in range(n_blocks):
          lgbphs_array[b,j*n_bins:(j+1)*n_bins] = lbphs_blocks[b]
      elif self.m_split == 'wavelets':
        lgbphs_array[j,0:n_blocks*n_bins] = lbphs_blocks.flatten()
      elif self.m_split == 'both':
        for b in range(n_bins):
          lgbphs_array[j*n_blocks + b,0:n_bins] = lbphs_blocks[b]
    
    # return the concatenated list of all histograms
    return lgbphs_array

    
