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
      mask_sigma = None,         # The sigma for random values areas outside image
      mask_neighbors = 5,        # The number of neighbors to consider while extrapolating
      mask_seed = None,          # The seed for generating random values during extrapolation
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

    mask_sigma
      Fill the area outside of image boundaries with random pixels from the border, by adding noise to the pixel values.
      To disable extrapolation, set this value to None.
      To disable adding random noise, set it to a negative value or 0.

    mask_neighbors
      The number of neighbors used during mask extrapolation.
      See :py:func:`bob.ip.base.extrapolate_mask` for details.

    mask_seed
      The random seed to apply for mask extrapolation.

      .. warning::
         When run in parallel, the same random seed will be applied to all parallel processes.
         Hence, results of parallel execution will differ from the results in serial execution.

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
        mask_sigma = mask_sigma,
        mask_neighbors = mask_neighbors,
        mask_seed = mask_seed,
        **kwargs
    )

    self.m_cropped_image_size = cropped_image_size
    self.m_cropped_positions = cropped_positions
    self.m_fixed_postions = fixed_positions
    self.m_color_channel = color_channel
    self.m_offset = offset
    self.m_supported_annotations = supported_annotations if supported_annotations is not None else (('reye', 'leye'), ('eye', 'mouth'))
    self.m_mask_sigma = mask_sigma
    self.m_mask_neighbors = mask_neighbors
    self.m_mask_rng = bob.core.random.mt19937(mask_seed) if mask_seed is not None else bob.core.random.mt19937()

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
    if pair[0] not in self.m_cropped_positions or pair[1] not in self.m_cropped_positions:
      import ipdb; ipdb.set_trace()
      raise KeyError("The given positions '%s' or '%s' are not found in the list of cropped positions: %s" % (pair[0], pair[1], self.m_cropped_positions))

    if key not in self.m_croppers:
      # generate cropper on the fly
      cropper = bob.ip.base.FaceEyesNorm(
        crop_size = (int(self.m_cropped_image_size[0] + 2 * self.m_offset), # cropped image height
                     int(self.m_cropped_image_size[1] + 2 * self.m_offset)), # cropped image width
        right_eye = (self.m_cropped_positions[pair[0]][0] + self.m_offset, # Y of first position (usually: right eye)
                     self.m_cropped_positions[pair[0]][1] + self.m_offset), # X of first position (usually: right eye)
        left_eye =  (self.m_cropped_positions[pair[1]][0] + self.m_offset,  # Y of second position (usually: left eye)
                     self.m_cropped_positions[pair[1]][1] + self.m_offset)   # X of second position (usually: left eye)
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

    keys = None
    # check, which type of annotations we have
    if self.m_fixed_postions:
      # get the cropper for the fixed positions
      keys = self.m_fixed_postions.keys()
      # take the fixed annotations
      annotations = self.m_fixed_postions
    if annotations:
      # get cropper for given annotations
      for pair in self.m_supported_annotations:
        if pair[0] in annotations and pair[1] in annotations:
          keys = pair
      if keys is None:
        raise ValueError("The given annoations '%s' did not contain the supported annotations '%s'" % (annotations, self.m_supported_annotations))
    else:
      # No annotations and no fixed positions: don't do any processing
      return image.astype(numpy.float64)

    cropper = self.__cropper__(keys)
    mask = self.__mask__(image.shape)

    # assure that the image is initialized with 0
    self.m_cropped_image[:] = 0.

    # perform the cropping
    cropper(
        image,  # input image
        mask,   # full input mask
        self.m_cropped_image, # cropped image
        self.m_cropped_mask,  # cropped mask
        right_eye = annotations[keys[0]], # position of first annotation, usually left eye
        left_eye = annotations[keys[1]]  # position of second annotation, usually right eye
    )

    if self.m_mask_sigma is not None:
      # extrapolate the mask so that pixels outside of the image original image region are filled with border pixels
      bob.ip.base.extrapolate_mask(self.m_cropped_mask, self.m_cropped_image, self.m_mask_sigma, self.m_mask_neighbors, self.m_mask_rng)

    return self.m_cropped_image


  def __call__(self, image, annotations = None):
    """Reads the input image, normalizes it according to the eye positions, and writes the resulting image"""
    return self.crop_face(image, annotations)
