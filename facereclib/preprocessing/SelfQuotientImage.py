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

import bob.ip.base
import numpy
import math
from .. import utils

from .FaceCrop import FaceCrop

class SelfQuotientImage (FaceCrop):
  """Crops the face according to the eye positions (if given), computes the self quotient image."""

  def __init__(self, sigma = 2., **kwargs):
    # call base class function
    FaceCrop.__init__(self, sigma = sigma, **kwargs)

    size = max(1, int(3. * sigma))
    self.m_self_quotient = bob.ip.base.SelfQuotientImage(size_min = size, sigma = sigma)

    if self.m_perform_image_cropping:
      self.m_self_quotient_image = numpy.ndarray(self.m_cropped_image.shape, numpy.float64)
    else:
      self.m_self_quotient_image = None


  def self_quotient(self, image):
    # create image in desired shape, if necessary
    if self.m_self_quotient_image is None or self.m_self_quotient_image.shape != image.shape:
      self.m_self_quotient_image = numpy.ndarray(image.shape, numpy.float64)

    # perform Tan&Triggs normalization
    self.m_self_quotient(image, self.m_self_quotient_image)

    return self.m_self_quotient_image


  def __call__(self, image, annotations = None):
    """Reads the input image, normalizes it according to the eye positions, computes the self quotient image, and writes the resulting image"""
    # crop the face using the base class method
    image = self.crop_face(image, annotations)

    # compute self quotient image
    self_quotient_image = self.self_quotient(image)

    if self.m_perform_image_cropping and annotations != None:
      # set the positions that were masked during face cropping to 0
      self_quotient_image[self.m_cropped_mask == False] = 0.

    # save the image to file
    return self_quotient_image
