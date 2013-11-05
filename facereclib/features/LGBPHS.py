#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy
import math

from .Extractor import Extractor
from .. import utils

class LGBPHS (Extractor):
  """Extractor for local Gabor binary pattern histogram sequences"""

  def __init__(
      self,
      # Block setup
      block_size,    # one or two parameters for block size
      block_overlap = 0, # one or two parameters for block overlap
      # Gabor parameters
      gabor_directions = 8,
      gabor_scales = 5,
      gabor_sigma = 2. * math.pi,
      gabor_maximum_frequency = math.pi / 2.,
      gabor_frequency_step = math.sqrt(.5),
      gabor_power_of_k = 0,
      gabor_dc_free = True,
      use_gabor_phases = False,
      # LBP parameters
      lbp_radius = 2,
      lbp_neighbor_count = 8,
      lbp_uniform = True,
      lbp_circular = True,
      lbp_rotation_invariant = False,
      lbp_compare_to_average = False,
      lbp_add_average = False,
      # histogram options
      sparse_histogram = False,
      split_histogram = None
  ):
    """Initializes the local Gabor binary pattern histogram sequence tool chain with the given file selector object"""

    # call base class constructor
    Extractor.__init__(
        self,

        block_size = block_size,
        block_overlap = block_overlap,
        gabor_directions = gabor_directions,
        gabor_scales = gabor_scales,
        gabor_sigma = gabor_sigma,
        gabor_maximum_frequency = gabor_maximum_frequency,
        gabor_frequency_step = gabor_frequency_step,
        gabor_power_of_k = gabor_power_of_k,
        gabor_dc_free = gabor_dc_free,
        use_gabor_phases = use_gabor_phases,
        lbp_radius = lbp_radius,
        lbp_neighbor_count = lbp_neighbor_count,
        lbp_uniform = lbp_uniform,
        lbp_circular = lbp_circular,
        lbp_rotation_invariant = lbp_rotation_invariant,
        lbp_compare_to_average = lbp_compare_to_average,
        lbp_add_average = lbp_add_average,
        sparse_histogram = sparse_histogram,
        split_histogram = split_histogram
    )

    # block parameters
    self.m_block_size = block_size if isinstance(block_size, (tuple, list)) else (block_size, block_size)
    self.m_block_overlap = block_overlap if isinstance(block_overlap, (tuple, list)) else (block_overlap, block_overlap)
    if self.m_block_size[0] < self.m_block_overlap[0] or self.m_block_size[1] < self.m_block_overlap[1]:
      raise ValueError("The overlap is bigger than the block size. This won't work. Please check your setup!")

    # Gabor wavelet transform class
    self.m_gwt = bob.ip.GaborWaveletTransform(
        number_of_scales = gabor_scales,
        number_of_angles = gabor_directions,
        sigma = gabor_sigma,
        k_max = gabor_maximum_frequency,
        k_fac = gabor_frequency_step,
        pow_of_k = gabor_power_of_k,
        dc_free = gabor_dc_free
    )
    self.m_trafo_image = None
    self.m_use_phases = use_gabor_phases


    # Initializes LBPHS processor
    real_h = self.m_block_size[0] + 2 * lbp_radius
    real_w = self.m_block_size[1] + 2 * lbp_radius
    real_oy = self.m_block_overlap[0] + 2 * lbp_radius
    real_ox = self.m_block_overlap[1] + 2 * lbp_radius

    self.m_lgbphs_extractor = bob.ip.LBPHSFeatures(
          block_h = real_h,
          block_w = real_w,
          overlap_h = real_oy,
          overlap_w = real_ox,
          lbp_radius = float(lbp_radius),
          lbp_neighbours = lbp_neighbor_count,
          circular = lbp_circular,
          to_average = lbp_compare_to_average,
          add_average_bit = lbp_add_average,
          uniform = lbp_uniform,
          rotation_invariant = lbp_rotation_invariant
    )

    self.m_split = split_histogram
    self.m_sparse = sparse_histogram
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

  def __call__(self, image):
    """Extracts the local Gabor binary pattern histogram sequence from the given image"""
    # perform GWT on image
    if self.m_trafo_image is None or self.m_trafo_image.shape[1:2] != image.shape:
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
    return utils.histogram.sparsify(lgbphs_array) if self.m_sparse else lgbphs_array

