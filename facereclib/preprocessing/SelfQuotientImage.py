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

from .FaceCrop import FaceCrop

class SelfQuotientImage (FaceCrop):
  """Crops the face according to the eye positions (if given), computes the self quotient image."""

  def __init__(self, config):
    # call base class function
    FaceCrop.__init__(self, config)

    size = max(1, int(3. * math.sqrt(config.sigma)))
    self.m_self_quotient = bob.ip.SelfQuotientImage(size_min = size, sigma2 = config.VARIANCE)
    self.m_self_quotient_image = numpy.ndarray(self.m_image.shape, numpy.float64)

  def self_quotient(self, image):
    # create image in desired shape, if necessary
    if self.m_self_quotient_image.shape != image.shape:
      self.m_self_quotient_image = numpy.ndarray(image.shape, numpy.float64)

    # perform Tan&Triggs normalization
    self.m_self_quotient(image, self.m_self_quotient_image)

    return self.m_self_quotient_image


  def __call__(self, input_file, output_file, annotations = None):
    """Reads the input image, normalizes it according to the eye positions, computes the self quotient image, and writes the resulting image"""
    # crop the face using the base class method
    image = self.crop_face(input_file, annotations)

    # compute self quotient image
    self_quotient_image = self.self_quotient(image)

    if annotations != None:
      # set the positions that were masked during face cropping to 0
      self_quotient_image[self.m_mask == False] = 0.

    # save the image to file
    bob.io.save(self_quotient_image, output_file)
