#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import bob.ip.base
import numpy
from .. import utils

from .Extractor import Extractor

class SIFTBobKeypoints(Extractor):
  """Extracts Sift descriptors according to the given keypoints"""

  def __init__(self,
      sigmas,
      height,
      width,
      n_octaves,
      n_scales,
      octave_min,
      sigma_n,
      sigma0,
      contrast_thres,
      edge_thres,
      norm_thres,
      kernel_radius_factor,
      set_sigma0_no_init_smoothing):

    # call base class constructor
    Extractor.__init__(self)

    # prepare SIFT extractor
    self.m_sigmas = sigmas
    self.m_height = height
    self.m_width = width
    self.m_n_octaves = n_octaves
    self.m_n_scales = n_scales
    self.m_octave_min = octave_min
    self.m_sigma_n = sigma_n
    self.m_sigma0 = sigma0
    self.m_contrast_thres = contrast_thres
    self.m_edge_thres = edge_thres
    self.m_norm_thres = norm_thres
    self.m_kernel_radius_factor = kernel_radius_factor
    self.m_set_sigma0_no_init_smoothing = set_sigma0_no_init_smoothing
    self.m_len_keypoint = len(self.m_sigmas)

    # SIFT extractor
    self.m_sift_extract = bob.ip.base.SIFT((self.m_height, self.m_width), self.m_n_octaves, self.m_n_scales, self.m_octave_min, self.m_sigma_n, self.m_sigma0, self.m_contrast_thres, self.m_edge_thres, self.m_norm_thres, self.m_kernel_radius_factor)
    if self.m_set_sigma0_no_init_smoothing: self.m_sift_extract.set_sigma0_no_init_smoothing()

  def __linearize__(self, descr):
    return numpy.reshape(descr, descr.size)

  def read(self, filename):
    """Read image and annotations stored in an HDF5 file"""
    f = bob.io.base.HDF5File(str(filename))
    image = f.read('image')
    annotations = f.read('annotations')
    return (image, annotations)


  def __call__(self, image):
    """Extract SIFT features given the image and the keypoints"""
    # Creates keypoints
    kp = []
    annotations = image[1]
    for k in range(annotations.shape[0]):
      for x in range(len(self.m_sigmas)):
        kp.append(bob.ip.base.GSSKeypoint(self.m_sigmas[x], annotations[k]))

    # Extracts and returns descriptors
    return self.__linearize__(self.m_sift_extract.compute_descriptor(image[0], kp))

