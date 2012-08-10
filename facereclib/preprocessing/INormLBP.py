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
from .FaceCrop import FaceCrop

class INormLBP (FaceCrop):
  """Crops the face according to the eye positions (if given), and performs I-Norm LBP on the resulting image"""

  def __init__(self, config):
    # call base class constructor
    FaceCrop.__init__(self, config)

    # overwrite base class functions to add radius of the LBP operator
    offset = config.OFFSET + config.RADIUS
    real_h = config.CROP_H + 2 * offset
    real_w = config.CROP_W + 2 * offset
    self.m_frontal_norm = bob.ip.FaceEyesNorm(config.CROP_EYES_D, real_h, real_w, config.CROP_OH + offset, config.CROP_OW + offset)
    if hasattr(config, 'MOUTH_POS'):
      self.m_profile_norm = bob.ip.FaceEyesNorm(real_h, real_w, config.EYE_POS[0] + offset, config.EYE_POS[1] + offset, config.MOUTH_POS[0] + offset, config.MOUTH_POS[1] + offset)
    self.m_image = numpy.ndarray((real_h, real_w), numpy.float64)
    self.m_mask = numpy.ndarray((real_h, real_w), numpy.bool)

    # lbp extraction
    self.m_lgb_extractor = bob.ip.LBP8R(config.RADIUS, config.CIRCULAR, config.TO_AVERAGE, config.ADD_AVERAGE_BIT, config.UNIFORM, config.ROT_INV, 0)
    self.m_i_norm_image = numpy.ndarray((config.CROP_H + 2 * config.OFFSET, config.CROP_W + 2 * config.OFFSET), numpy.uint16)


  def i_norm(self, image):
    """Computes the I-Norm-LBP normalization on the given image"""
    # check the shape of the image and correct it if needed
    desired_shape = (image.shape[0] - 2*self.m_config.RADIUS, image.shape[1] - 2*self.m_config.RADIUS)
    if self.m_i_norm_image.shape != desired_shape:
      self.m_i_norm_image = numpy.ndarray(desired_shape, numpy.uint16)
      
    # perform normalization
    self.m_lgb_extractor(image, self.m_i_norm_image)
    
    return self.m_i_norm_image

  def __call__(self, input_file, output_file, annotations = None):
    """Reads the input image, normalizes it according to the eye positions, computes I-Norm-LBP's, and writes the resulting image"""
    # crop the face using the base class method
    image = self.crop_face(input_file, annotations)
    
    # compute I-Norm-LBP image
    i_norm_image = self.i_norm(image)
    
    if annotations != None:
      # set the positions that were masked during face cropping to 0; respect the size change of the two images!
      # I am not sure if 0 is the right value here...
      R = self.m_config.RADIUS
      i_norm_image[self.m_mask[R:-R,R:-R] == False] = 0

    # save the image to file
    bob.io.save(i_norm_image.astype(numpy.float64), output_file)

