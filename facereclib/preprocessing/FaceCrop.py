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
import math
from .. import utils

class FaceCrop:
  """Crops the face according to the eye positions"""

  def __init__(self, config):
    self.m_config = config
    self.m_color_channel = config.COLOR_CHANNEL if hasattr(config, 'COLOR_CHANNEL') else 'gray'
    # prepare image normalization
    offset = config.OFFSET
    real_h = config.CROPPED_IMAGE_HEIGHT + 2 * offset
    real_w = config.CROPPED_IMAGE_WIDTH + 2 * offset
    self.m_frontal_norm = bob.ip.FaceEyesNorm(real_h, real_w, config.RIGHT_EYE_POS[0] + offset, config.RIGHT_EYE_POS[1] + offset, config.LEFT_EYE_POS[0] + offset, config.LEFT_EYE_POS[1] + offset)
    if hasattr(config, 'MOUTH_POS'):
      self.m_profile_norm = bob.ip.FaceEyesNorm(real_h, real_w, config.EYE_POS[0] + offset, config.EYE_POS[1] + offset, config.MOUTH_POS[0] + offset, config.MOUTH_POS[1] + offset)
    self.m_image = numpy.ndarray((real_h, real_w), numpy.float64)
    self.m_mask = numpy.ndarray((real_h, real_w), numpy.bool)


  def crop_face(self, image, annotations):
    """Executes the face cropping on the given image and returns the cropped version of it"""
    # convert to the desired color channel
    image = utils.gray_channel(image, self.m_color_channel)

    if hasattr(self.m_config, 'FIXED_RIGHT_EYE') and hasattr(self.m_config, 'FIXED_LEFT_EYE') or hasattr(self.m_config, 'FIXED_EYE') and hasattr(self.m_config, 'FIXED_MOUTH'):
      # use the fixed eye positions to perform normalization
      if annotations == None or ('leye' in annotations and 'reye' in annotations):
        assert hasattr(self.m_config, 'FIXED_RIGHT_EYE') and hasattr(self.m_config, 'FIXED_LEFT_EYE')
        # use the frontal normalizer
        right = self.m_config.FIXED_RIGHT_EYE
        left = self.m_config.FIXED_LEFT_EYE
        self.m_frontal_norm(image, mask, self.m_image, self.m_mask, rigth[0], right[1], leftz[0], left[1])
      else:
        assert hasattr(self.m_config, 'FIXED_EYE') and hasattr(self.m_config, 'FIXED_MOUTH')
        # use profile normalization
        eye = self.m_config.FIXED_EYE
        mouth = self.m_config.FIXED_MOUTH
        self.m_profile_norm(image, mask, self.m_image, self.m_mask, eye[0], eye[1], mouth[0], mouth[1])

    elif annotations == None:
      # simply return the image
      return image.astype(numpy.float64)
    else:

      assert ('leye' in annotations and 'reye' in annotations) or ('eye' in annotations and 'mouth' in annotations)
      mask = numpy.ndarray(image.shape, numpy.bool)
      mask.fill(True)
      if 'leye' in annotations and 'reye' in annotations:
        # use the frontal normalizer
        self.m_frontal_norm(image, mask, self.m_image, self.m_mask, annotations['reye'][0], annotations['reye'][1], annotations['leye'][0], annotations['leye'][1])
      else:
        # use profile normalization
        self.m_profile_norm(image, mask, self.m_image, self.m_mask, annotations['eye'][0], annotations['eye'][1], annotations['mouth'][0], annotations['mouth'][1])

      # assure that pixels from the masked area are 0
      self.m_image[self.m_mask == False] = 0.

      return self.m_image


  def __call__(self, image, annotations = None):
    """Reads the input image, normalizes it according to the eye positions, and writes the resulting image"""
    return self.crop_face(image, annotations)
