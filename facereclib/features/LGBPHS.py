#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.ip.gabor
import bob.ip.base

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
    self.m_gwt = bob.ip.gabor.Transform(
        number_of_scales = gabor_scales,
        number_of_directions = gabor_directions,
        sigma = gabor_sigma,
        k_max = gabor_maximum_frequency,
        k_fac = gabor_frequency_step,
        power_of_k = gabor_power_of_k,
        dc_free = gabor_dc_free
    )
    self.m_trafo_image = None
    self.m_use_phases = use_gabor_phases

    self.m_lbp = bob.ip.base.LBP(
        neighbors = lbp_neighbor_count,
        radius = float(lbp_radius),
        circular = lbp_circular,
        to_average = lbp_compare_to_average,
        add_average_bit = lbp_add_average,
        uniform = lbp_uniform,
        rotation_invariant = lbp_rotation_invariant,
        border_handling = 'wrap'
    )

    self.m_split = split_histogram
    self.m_sparse = sparse_histogram
    if self.m_sparse and self.m_split:
      raise ValueError("Sparse histograms cannot be split! Check your setup!")


  def __fill__(self, lgbphs_array, lgbphs_blocks, j):
    """Copies the given array into the given blocks"""
    # fill array in the desired shape
    if self.m_split is None:
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
    if self.m_trafo_image is None or self.m_trafo_image.shape[1:3] != image.shape:
      # create trafo image
      self.m_trafo_image = numpy.ndarray((self.m_gwt.number_of_wavelets, image.shape[0], image.shape[1]), numpy.complex128)

    # perform Gabor wavelet transform
    self.m_gwt.transform(image, self.m_trafo_image)

    jet_length = self.m_gwt.number_of_wavelets * (2 if self.m_use_phases else 1)

    lgbphs_array = None
    # iterate through the layers of the trafo image
    for j in range(self.m_gwt.number_of_wavelets):
      # compute absolute part of complex response
      abs_image = numpy.abs(self.m_trafo_image[j])
      # Computes LBP histograms
      abs_blocks = bob.ip.base.lbphs(abs_image, self.m_lbp, self.m_block_size, self.m_block_overlap)

      # Converts to Blitz array (of different dimensionalities)
      self.m_n_bins = abs_blocks.shape[1]
      self.m_n_blocks = abs_blocks.shape[0]

      if self.m_split is None:
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
      if lgbphs_array is None:
        lgbphs_array = numpy.ndarray(shape, 'float64')

      # fill the array with the absolute values of the Gabor wavelet transform
      self.__fill__(lgbphs_array, abs_blocks, j)

      if self.m_use_phases:
        # compute phase part of complex response
        phase_image = numpy.angle(self.m_trafo_image[j])
        # Computes LBP histograms
        phase_blocks = bob.ip.base.lbphs(phase_image, self.m_lbp, self.m_block_size, self.m_block_overlap)
        # fill the array with the phases at the end of the blocks
        self.__fill__(lgbphs_array, phase_blocks, j + self.m_gwt.number_of_wavelets)


    # return the concatenated list of all histograms
    return utils.histogram.sparsify(lgbphs_array) if self.m_sparse else lgbphs_array

