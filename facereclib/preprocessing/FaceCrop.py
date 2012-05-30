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

class FaceCrop:
  """Crops the face according to the eye positions"""

  def __init__(self, config):
    self.m_config = config
    # prepare image normalization
    self.m_fen = bob.ip.FaceEyesNorm(config.CROP_EYES_D, config.CROP_H, config.CROP_W, config.CROP_OH, config.CROP_OW)
    self.m_fen_image = numpy.ndarray((config.CROP_H, config.CROP_W), numpy.float64) 

  def __call__(self, input_file, output_file, eye_pos = None):
    """Reads the input image, normalizes it according to the eye positions, and writes the resulting image"""
    image = bob.io.load(str(input_file))
    # convert to grayscale
    if(image.ndim == 3):
      image = bob.ip.rgb_to_gray(image)

    if eye_pos == None:
      # simply save the image to file
      bob.io.save(image, output_file)
    else:
      # perform image normalization
      self.m_fen(image, self.m_fen_image, eye_pos[1], eye_pos[0], eye_pos[3], eye_pos[2])
      # save output image    
      bob.io.save(self.m_fen_image, output_file)


class StaticFaceCrop:
  """Crops the face according to FIXED eye positions"""
  def __init__(self, config):
    self.m_config = config
    # prepare image normalization
    self.m_fen = bob.ip.FaceEyesNorm(config.CROP_EYES_D, config.CROP_H, config.CROP_W, config.CROP_OH, config.CROP_OW)
    self.m_fen_image = numpy.ndarray((config.CROP_H, config.CROP_W), numpy.float64) 

  def __call__(self, input_file, output_file, eye_pos = None):
    """Reads the input image, normalizes it according to the eye positions, and writes the resulting image"""
    image = bob.io.load(str(input_file))
    # convert to grayscale
    if(image.ndim == 3):
      image = bob.ip.rgb_to_gray(image)

    # simply save the image to file
    self.m_fen(image, self.m_fen_image, self.m_config.RIGHT_EYE[0], self.m_config.RIGHT_EYE[1], self.m_config.LEFT_EYE[0], self.m_config.LEFT_EYE[1])
    # save output image    
    bob.io.save(self.m_fen_image, output_file)

