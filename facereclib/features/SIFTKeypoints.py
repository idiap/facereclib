#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import bob.ip.base

import numpy

from .Extractor import Extractor

class SIFTKeypoints (Extractor):
  """Extracts Sift descriptors according to the given keypoints"""

  # TODO: REFACTOR: assign better names for variables
  def __init__(
      self,
      sigmas,
      height,
      width,
      n_intervals,
      n_octaves,
      octave_min,
      edge_thres,
      peak_thres,
      magnif,
      estimate_orientation = False
  ):

    if not hasattr(bob.ip.base, "VLSIFT"):
      raise NotImplementedError("VLSIFT is not part of bob.ip.base; maybe SIFT headers aren't installed in your system? Try out facereclib.features.SIFTBobKeypoints!")

    # call base class constructor
    Extractor.__init__(self)

    # prepare SIFT extractor
    self.m_n_scales = len(sigmas)
    self.m_sigmas = numpy.ndarray(shape=(self.m_n_scales,), dtype=numpy.float64)
    # TODO: restructure this code
    if self.m_n_scales >= 5: self.m_sigmas[4] = sigmas[4]
    if self.m_n_scales >= 4: self.m_sigmas[3] = sigmas[3]
    if self.m_n_scales >= 3: self.m_sigmas[2] = sigmas[2]
    if self.m_n_scales >= 2: self.m_sigmas[1] = sigmas[1]
    if self.m_n_scales >= 1: self.m_sigmas[0] = sigmas[0]
    self.m_estimate_orientation = estimate_orientation
    if(estimate_orientation): self.m_len_keypoint = 3
    else: self.m_len_keypoint = 4
    self.m_height = height
    self.m_width = width
    self.m_n_intervals = n_intervals
    self.m_n_octaves = n_octaves
    self.m_octave_min = octave_min
    self.m_peak_thres = peak_thres
    self.m_edge_thres = edge_thres
    self.m_magnif = magnif
    # SIFT extractor
    self.m_sift_extract = bob.ip.base.VLSIFT((self.m_height, self.m_width), self.m_n_intervals, self.m_n_octaves, self.m_octave_min, self.m_peak_thres, self.m_edge_thres, self.m_magnif)

  def __linearize_cut__(self, descr):
    l_vec = 128 # Length of the SIFT descriptors
    l_full = len(descr) * l_vec
    output = numpy.ndarray(shape=(l_full,), dtype=descr[0].dtype)
    k=0
    for vec in descr:
      output[k*l_vec:(k+1)*l_vec] = vec[4:132] # Cut first 4 values
      k=k+1
    return output

  def __call__(self, img_annots):
    """Extract SIFT features given the image and the keypoints"""
    image = img_annots[0]
    annotations = img_annots[1]

    # Creates keypoints numpy array
    kp = numpy.ndarray(shape=(len(annotations)*self.m_n_scales,self.m_len_keypoint), dtype=numpy.float64)
    c=0
    for k in range(annotations.shape[0]):
      for x in range(self.m_n_scales):
        if(self.m_estimate_orientation):
          kp[c*self.m_n_scales+x,:]=[annotations[k,0], annotations[k,1], self.m_sigmas[x]]
        else:
          kp[c*self.m_n_scales+x,:]=[annotations[k,0], annotations[k,1], self.m_sigmas[x], 0.]
      c=c+1

    # Extracts and returns descriptors
    return self.__linearize_cut__(self.m_sift_extract(image, kp))
