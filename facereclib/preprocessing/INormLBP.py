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

class INormLBP:
  """Crops the face according to the eye positions (if given), and performs histogram equalization on the resulting image"""

  def __init__(self, config):
    self.m_config = config
    self.m_color_channel = config.color_channel if hasattr(config, 'color_channel') else 'gray'
    # prepare image normalization; add two pixels of rim that will be cut off by the algorithm
    real_h = config.CROP_H + 2 * config.OFFSET
    real_w = config.CROP_W + 2 * config.OFFSET
    self.m_fen = bob.ip.FaceEyesNorm(config.CROP_EYES_D, real_h, real_w, config.CROP_OH + config.OFFSET, config.CROP_OW + config.OFFSET)
    self.m_fen_image = numpy.ndarray((real_h, real_w), numpy.float64) 
    # lbp extraction
    self.m_lgb_extractor = bob.ip.LBP8R(config.RADIUS, config.CIRCULAR, config.TO_AVERAGE, config.ADD_AVERAGE_BIT, config.UNIFORM, config.ROT_INV, 0)


  def __call__(self, input_file, output_file, eye_pos = None):
    """Reads the input image, normalizes it according to the eye positions, and writes the resulting image"""
    image = bob.io.load(str(input_file))
    # convert to grayscale
    image = utils.gray_channel(image, self.m_color_channel)

    if eye_pos == None:
      inorm_image = self.m_lgb_extractor(image)
    else:
      # perform image normalization
      self.m_fen(image, self.m_fen_image, eye_pos[1], eye_pos[0], eye_pos[3], eye_pos[2])
      inorm_image = self.m_lgb_extractor(self.m_fen_image)
      
    # simply save the image to file
    inorm_image = inorm_image.astype(numpy.float64)
    bob.io.save(inorm_image, output_file)

