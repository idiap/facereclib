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

class HistogramEqualization (FaceCrop):
  """Crops the face according to the eye positions (if given), and performs histogram equalization on the resulting image"""

  def __init__(self, config):
    # call base class constructor
    FaceCrop.__init__(self, config)
    # initialize histogram image
    self.m_histogram_image = numpy.ndarray(self.m_image.shape, numpy.float64)


  def equalize_histogram(self, image):
    """Calls bob function to do the actual histogram equalization"""
    bob.ip.histogram_equalization(image.astype(numpy.uint8), self.m_histogram_image)

    return self.m_histogram_image


  def __call__(self, input_file, output_file, annotations = None):
    """Reads the input image, normalizes it according to the eye positions, performs histogram equalization, and writes the resulting image"""
    # crop the face using the base class method
    image = self.crop_face(input_file, annotations)

    # perform histogram equalization
    histogram_image = self.equalize_histogram(image)

    if annotations != None:
      # set the positions that were masked during face cropping to 0
      histogram_image[self.m_mask == False] = 0.

    # save the image to file
    bob.io.save(histogram_image, output_file)

