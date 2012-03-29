#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

"""Image preprocessing tools"""

import bob
import numpy
import math

class NullPreprocessor:
  """Skips proprocessing files by simply copying the contents into an hdf5 file 
  (and perform gray scale conversion if required)"""
  def __init__(self, config):
    pass
    
  def __call__(self, input_file, eye_pos):
    image = bob.io.load(str(input_file))
    # convert to grayscale
    if(image.ndim == 3):
      image = bob.ip.rgb_to_gray(image)
    return image



class FaceCrop:
  """Croppes the face according to the eye positions"""

  def __init__(self, config):
    self.m_config = config
    # prepare image normalization
    self.m_fen = bob.ip.FaceEyesNorm(config.CROP_EYES_D, config.CROP_H, config.CROP_W, config.CROP_OH, config.CROP_OW)
    self.m_fen_image = numpy.ndarray((config.CROP_H, config.CROP_W), numpy.float64) 

  def __call__(self, input_file, eye_pos):
    """Reads the input image, normalizes it according to the eye positions, and writes the resulting image"""
    image = bob.io.load(str(input_file))
    # convert to grayscale
    if(image.ndim == 3):
      image = bob.ip.rgb_to_gray(image)
      
    # perform image normalization
    self.m_fen(image, self.m_fen_image, eye_pos[1], eye_pos[0], eye_pos[3], eye_pos[2])

    # return the normalized image    
    return self.m_fen_image
    
    
    
class TanTriggs:
  """Croppes the face and applies Tan-Triggs algorithm"""

  def __init__(self, config):
    self.m_config = config
    # prepare image normalization
    self.m_fen = bob.ip.FaceEyesNorm(config.CROP_EYES_D, config.CROP_H, config.CROP_W, config.CROP_OH, config.CROP_OW)
    self.m_fen_image = numpy.ndarray((config.CROP_H, config.CROP_W), numpy.float64) 
    self.m_tan = bob.ip.TanTriggs(config.GAMMA, config.SIGMA0, config.SIGMA1, config.SIZE, config.THRESHOLD, config.ALPHA)
    self.m_tan_image = numpy.ndarray((config.CROP_H, config.CROP_W), numpy.float64)

  def __call__(self, input_file, eye_pos):
    """Reads the input image, normalizes it according to the eye positions, and writes the resulting image"""
    image = bob.io.load(str(input_file))
    # convert to grayscale
    if(image.ndim == 3):
      image = bob.ip.rgb_to_gray(image)
      
    # perform image normalization
    self.m_fen(image, self.m_fen_image, eye_pos[1], eye_pos[0], eye_pos[3], eye_pos[2])
    
    # perform Tan-Triggs
    self.m_tan(self.m_fen_image, self.m_tan_image)
    
    # return the normalized image
    return self.m_tan_image
    
    

