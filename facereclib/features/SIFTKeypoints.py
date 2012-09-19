#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import bob
import numpy

class SIFTKeypoints:
  """Extracts Sift descriptors according to the given keypoints"""

  def __init__(self, config):
    self.m_config = config
    # prepare SIFT extractor
    self.m_n_scales = config.N_SCALES
    self.m_sigmas = numpy.ndarray(shape=(self.m_n_scales,), dtype=numpy.float64)
    if self.m_n_scales >= 5: self.m_sigmas[4] = config.SIGMA4
    if self.m_n_scales >= 4: self.m_sigmas[3] = config.SIGMA3
    if self.m_n_scales >= 3: self.m_sigmas[2] = config.SIGMA2
    if self.m_n_scales >= 2: self.m_sigmas[1] = config.SIGMA1
    if self.m_n_scales >= 1: self.m_sigmas[0] = config.SIGMA0
    self.m_estimate_orientation = config.ESTIMATE_ORIENTATION
    if(config.ESTIMATE_ORIENTATION): self.m_len_keypoint = 3
    else: self.m_len_keypoint = 4
    self.m_height = config.HEIGHT
    self.m_width = config.WIDTH
    self.m_n_intervals = config.N_INTERVALS
    self.m_n_octaves = config.N_OCTAVES
    self.m_octave_min = config.OCTAVE_MIN
    self.m_peak_thres = config.PEAK_THRES
    self.m_edge_thres = config.EDGE_THRES
    self.m_magnif = config.MAGNIF
    # SIFT extractor
    self.m_sift_extract = bob.ip.VLSIFT(self.m_height, self.m_width, self.m_n_intervals, self.m_n_octaves, self.m_octave_min, self.m_peak_thres, self.m_edge_thres, self.m_magnif)

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

