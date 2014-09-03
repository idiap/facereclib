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
from .. import utils
from .FaceCrop import FaceCrop

class HistogramEqualization (FaceCrop):
  """Crops the face according to the eye positions (if given), and performs histogram equalization on the resulting image"""

  def __init__(self, **kwargs):

    # call base class constructor with its set of parameters
    FaceCrop.__init__(self, **kwargs)

    if self.m_perform_image_cropping:
      # input image will be the output of the face cropper
      self.m_histogram_image = numpy.ndarray(self.m_cropped_image.shape, numpy.float64)
    else:
      # resolution of input image is not known yet
      self.m_histogram_image = None


  def equalize_histogram(self, image):
    """Calls bob function to do the actual histogram equalization"""
    # create image in desired shape, if necessary
    if self.m_histogram_image is None or self.m_histogram_image.shape != image.shape:
      self.m_histogram_image = numpy.ndarray(image.shape, numpy.float64)

    bob.ip.base.histogram_equalization(numpy.round(image).astype(numpy.uint8), self.m_histogram_image)

    return self.m_histogram_image


  def __call__(self, image, annotations = None):
    """Reads the input image, normalizes it according to the eye positions, performs histogram equalization, and writes the resulting image"""
    # crop the face using the base class method

    image = self.crop_face(image, annotations)

    # perform histogram equalization
    histogram_image = self.equalize_histogram(image)

    if self.m_perform_image_cropping and annotations != None:
      # set the positions that were masked during face cropping to 0
      histogram_image[self.m_cropped_mask == False] = 0.

    # save the image to file
    return histogram_image

