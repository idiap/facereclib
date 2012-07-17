#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date: Thu May 24 10:41:42 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import bob
import numpy
from .. import utils

class HistogramEqualization:
  """Crops the face according to the eye positions (if given), and performs histogram equalization on the resulting image"""

  def __init__(self, config):
    self.m_config = config
    self.m_color_channel = config.color_channel if hasattr(config, 'color_channel') else 'gray'
    # prepare image normalization
    self.m_fen = bob.ip.FaceEyesNorm(config.CROP_EYES_D, config.CROP_H, config.CROP_W, config.CROP_OH, config.CROP_OW)
    self.m_fen_image = numpy.ndarray((config.CROP_H, config.CROP_W), numpy.float64) 

  def __equalize_histogram__(self, image):
    image = image.astype(numpy.uint8)
    histogram = bob.ip.histogram(image, 255).astype(numpy.float64)
    histogram /= float(image.size)
  
    # compute cumulative histogram density function  
    cdf = [0.] * len(histogram)
    for i in range(1,len(histogram)):
      cdf[i] = cdf[i-1] + histogram[i]
    
    # normalize image
    he_image = numpy.ndarray(image.shape, numpy.float64)
    for y in range(image.shape[0]):
      for x in range(image.shape[1]):
        # Multiply with 255 to shift the normalized cdf values to pixel ranges 0..255
        he_image[y,x] = cdf[image[y,x]] * 255.
    
    return he_image
    

  def __call__(self, input_file, output_file, eye_pos = None):
    """Reads the input image, normalizes it according to the eye positions, and writes the resulting image"""
    image = bob.io.load(str(input_file))
    # convert to grayscale
    image = utils.gray_channel(image, self.m_color_channel)

    if eye_pos == None:
      he_image = self.__equalize_histogram__(image)
    else:
      # perform image normalization
      self.m_fen(image, self.m_fen_image, eye_pos[1], eye_pos[0], eye_pos[3], eye_pos[2])
      he_image = self.__equalize_histogram__(self.m_fen_image)
      
    # simply save the image to file
    bob.io.save(he_image, output_file)

