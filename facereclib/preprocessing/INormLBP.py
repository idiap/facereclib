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

class INormLBP (FaceCrop):
  """Crops the face according to the eye positions (if given), and performs I-Norm LBP on the resulting image"""

  def __init__(
      self,
      radius = 2,  # Radius of the LBP
      is_uniform = False, # use uniform LBP?
      is_circular = True, # use circular LBP?
      is_rotation_invariant = False,
      compare_to_average = False,
      add_average_bit = False,
      # Parameters of face cropping; need to be adapted, if set
      cropped_image_size = None,# resolution of the cropped image, in order (HEIGHT,WIDTH); if not given, no face cropping will be performed
      cropped_positions = None, # dictionary of the cropped positions, usually: {'reye':(RIGHT_EYE_Y, RIGHT_EYE_X) , 'leye':(LEFT_EYE_Y, LEFT_EYE_X)}
      **kwargs # remaining parameters of the face cropping
  ):

    self.m_radius = radius

    # call base class constructor
    FaceCrop.__init__(
        self,
        cropped_image_size = cropped_image_size,
        cropped_positions = cropped_positions,

        radius = radius,
        is_uniform = is_uniform,
        is_circular = is_circular,
        is_rotation_invariant = is_rotation_invariant,
        compare_to_average = compare_to_average,
        add_average_bit = add_average_bit,

        **kwargs
    )

    # lbp extraction
    self.m_lgb_extractor = bob.ip.base.LBP(8, radius, is_circular, compare_to_average, add_average_bit, is_uniform, is_rotation_invariant, 'regular', 'wrap')
    if self.m_perform_image_cropping:
      self.m_i_norm_image = numpy.ndarray(self.m_cropped_image.shape, numpy.uint16)
    else:
      self.m_i_norm_image = None



  def i_norm(self, image):
    """Computes the I-Norm-LBP normalization on the given image"""
    # check the shape of the image and correct it if needed
    desired_shape = image.shape
    if self.m_i_norm_image.shape != desired_shape:
      self.m_i_norm_image = numpy.ndarray(desired_shape, numpy.uint16)

    # perform normalization
    self.m_lgb_extractor(image, self.m_i_norm_image)

    return self.m_i_norm_image

  def __call__(self, image, annotations = None):
    """Reads the input image, normalizes it according to the eye positions, computes I-Norm-LBP's, and writes the resulting image"""
    # crop the face using the base class method
    image = self.crop_face(image, annotations)

    # compute I-Norm-LBP image
    i_norm_image = self.i_norm(image)

    if self.m_perform_image_cropping and annotations != None:
      # set the positions that were masked during face cropping to 0; respect the size change of the two images!
      # I am not sure if 0 is the right value here...
      i_norm_image[self.m_cropped_mask == False] = 0

    # save the image to file
    return i_norm_image.astype(numpy.float64)

