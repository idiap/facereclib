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
from .Preprocessor import Preprocessor

class FaceCrop (Preprocessor):
  """Crops the face according to the eye positions"""

  def __init__(
      self,
      cropped_image_size = None, # resolution of the cropped image, in order (HEIGHT,WIDTH); if not given, no face cropping will be performed
      cropped_positions = None,  # dictionary of the cropped positions, usually: {'reye':(RIGHT_EYE_Y, RIGHT_EYE_X) , 'leye':(LEFT_EYE_Y, LEFT_EYE_X)}
      fixed_positions = None,    # dictionary of FIXED positions in the original image; if specified, annotations from the database will be ignored
      color_channel = 'gray',    # the color channel to extract from colored images, if colored images are in the database
      offset = 0,                # if your feature extractor requires a specific offset, you might want to specify it here
      supported_annotations = None, # The set of annotations that this cropper excepts; (('reye', 'leye'), ('eye', 'mouth')) by default
      **kwargs                   # parameters to be written in the __str__ method
  ):
    """Parameters of the constructor of this preprocessor:

    cropped_image_size
      The size of the resulting cropped images.

    cropped_positions
      The coordinates in the cropped image, where the annotated points should be put to.
      This parameter is a dictionary with usually two elements, e.g., {'reye':(RIGHT_EYE_Y, RIGHT_EYE_X) , 'leye':(LEFT_EYE_Y, LEFT_EYE_X)}.

    fixed_positions
      If specified, ignore the annotations from the database and use these fixed positions throughout.

    color_channel
      In case of color images, which color channel should be used?

    offset
      An offset for feature extraction; will affect the cropped_image_size and the cropped_image_positions

    supported_annotations
      A list of supported pairs of annotations.
      If the database has different names for the annotations, they should be put here (used mainly for testing purposes).
      If not specified, (('reye', 'leye'), ('eye', 'mouth')) is used.

    """

    # call base class constructor
    Preprocessor.__init__(
        self,
        cropped_image_size = cropped_image_size,
        cropped_positions = cropped_positions,
        fixed_positions = fixed_positions,
        color_channel = color_channel,
        offset = offset,
        supported_annotations = supported_annotations,
        **kwargs
    )

    self.m_cropped_image_size = cropped_image_size
    self.m_cropped_positions = cropped_positions
    self.m_fixed_postions = fixed_positions
    self.m_color_channel = color_channel
    self.m_offset = offset
    self.m_supported_annotations = supported_annotations if supported_annotations is not None else (('reye', 'leye'), ('eye', 'mouth'))

    if fixed_positions:
      assert len(fixed_positions) == 2

    # define our set of functions
    self.m_croppers = {}
    self.m_original_masks = {}

    self.m_perform_image_cropping = self.m_cropped_image_size is not None

    if self.m_perform_image_cropping:
      # define the preprocessed image once
      self.m_cropped_image = numpy.ndarray((self.m_cropped_image_size[0] + 2 * self.m_offset, self.m_cropped_image_size[1] + 2 * self.m_offset), numpy.float64)
      # define the mask; this mask can be used in derived classes to further process the image
      self.m_cropped_mask = numpy.ndarray(self.m_cropped_image.shape, numpy.bool)

  def __cropper__(self, pair):
    key = (pair[0] + "+" + pair[1])
    assert pair[0] in self.m_cropped_positions and pair[1] in self.m_cropped_positions

    if key not in self.m_croppers:
      # generate cropper on the fly
      cropper = bob.ip.FaceEyesNorm(
          self.m_cropped_image_size[0] + 2 * self.m_offset, # cropped image height
          self.m_cropped_image_size[1] + 2 * self.m_offset, # cropped image width
          self.m_cropped_positions[pair[0]][0] + self.m_offset, # Y of first position (usually: right eye)
          self.m_cropped_positions[pair[0]][1] + self.m_offset, # X of first position (usually: right eye)
          self.m_cropped_positions[pair[1]][0] + self.m_offset,  # Y of second position (usually: left eye)
          self.m_cropped_positions[pair[1]][1] + self.m_offset   # X of second position (usually: left eye)
      )
      self.m_croppers[key] = cropper

    # return cropper for this type
    return self.m_croppers[key]

  def __mask__(self, shape):
    key = (str(shape[0]) + "x" + str(shape[1]))
    if key not in self.m_original_masks:
      # generate mask for the given image resolution
      mask = numpy.ndarray(shape, numpy.bool)
      mask.fill(True)
      self.m_original_masks[key] = mask
    # return the stored mask for the given resolution
    return self.m_original_masks[key]


  def crop_face(self, image, annotations):
    """Executes the face cropping on the given image and returns the cropped version of it"""
    # convert to the desired color channel
    image = utils.gray_channel(image, self.m_color_channel)

    if not self.m_perform_image_cropping:
      return image

    # check, which type of annotations we have
    if self.m_fixed_postions:
      # get the cropper for the fixed positions
      keys = sorted(self.m_fixed_postions.keys())
      # take the fixed annotations
      annotations = self.m_fixed_postions
    elif annotations:
      # get cropper for given annotations
      for pair in self.m_supported_annotations:
        if pair[0] in annotations and pair[1] in annotations:
          keys = pair
    else:
      # No annotations and no fixed positions: don't do any processing
      return image.astype(numpy.float64)

    cropper = self.__cropper__(keys)
    mask = self.__mask__(image.shape)

    # perform the cropping
    cropper(
        image,  # input image
        mask,   # full input mask
        self.m_cropped_image, # cropped image
        self.m_cropped_mask,  # cropped mase
        annotations[keys[0]][0], # Y-position of first annotation, usually left eye
        annotations[keys[0]][1], # X-position of first annotation, usually left eye
        annotations[keys[1]][0], # Y-position of first annotation, usually right eye
        annotations[keys[1]][1]  # X-position of first annotation, usually right eye
    )

    # assure that pixels from the masked area are 0
    self.m_cropped_image[self.m_cropped_mask == False] = 0.

    return self.m_cropped_image


  def __call__(self, image, annotations = None):
    """Reads the input image, normalizes it according to the eye positions, and writes the resulting image"""
    return self.crop_face(image, annotations)
