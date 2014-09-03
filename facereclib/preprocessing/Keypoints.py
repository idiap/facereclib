#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# @date: Thu Aug 12 18:58:00 CEST 2012
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

import bob.io.base

import numpy
import math
from .. import utils
from .Preprocessor import Preprocessor
from .FaceCrop import FaceCrop

class Keypoints (FaceCrop):
  """Extracts keypoints (from a possibly normalized/cropped face image)"""

  def __init__(
      self,
      crop_image = False, # Tells if the image is cropped before extracting keypoints or not
      color_channel = 'gray', # The color channel to keep
      fixed_annotations=None, # Fixed annotations used to all images. If set, this should be a dictionary of pairs (y,x) (or (dy,dx) if relative_annotations is set)
      cropped_domain_annotations=False, # If sets to true, the 'fixed' annotations are assumed to be in the cropped image (in the 'original' image otherwise)
      relative_annotations=False, # Relative annotations in ratios wrt. to the middle of the eyes points and in 'eye distance' unit
      use_eye_corners=False,
      **kwargs
  ): # Assume that eye corners are used instead of eye centers!
    # call base class constructor
    FaceCrop.__init__(self, **kwargs)
    self.m_color_channel = color_channel
    self.m_crop_image = crop_image
    self.m_fixed_annotations = fixed_annotations
    self.m_cropped_domain_annotations = cropped_domain_annotations
    self.m_relative_annotations = relative_annotations
    self.m_use_eye_corners = use_eye_corners

  def _distance(self, y1, x1, y2, x2):
    return math.sqrt((y1-y2)**2+(x1-x2)**2)

  def _get_eyes_coordinates(self, annotations):
    # Make sure the required annotations are here
    if self.m_use_eye_corners == True:
      keys = ('reyei', 'reyeo', 'leyei', 'leyeo')
    else:
      keys = self.m_supported_annotations[0]
    for k in keys: assert k in annotations

    # Eyes coordinates in the image
    if self.m_use_eye_corners == True:  # Takes the mean coordinates of the eye corners as the ones of the eye centers
      c_reye_y = (annotations[keys[0]][0] + annotations[keys[1]][0]) / 2.
      c_reye_x = (annotations[keys[0]][1] + annotations[keys[1]][1]) / 2.
      c_leye_y = (annotations[keys[2]][0] + annotations[keys[3]][0]) / 2.
      c_leye_x = (annotations[keys[2]][1] + annotations[keys[3]][1]) / 2.
    else: # eye centers coordinates are already there
      c_reye_y = annotations[keys[0]][0] # Y-position of first annotation, usually right eye
      c_reye_x = annotations[keys[0]][1] # X-position of first annotation, usually right eye
      c_leye_y = annotations[keys[1]][0] # Y-position of second annotation, usually left eye
      c_leye_x = annotations[keys[1]][1] # X-position of second annotation, usually left eye
    return (c_reye_y, c_reye_x, c_leye_y, c_leye_x)

  def _compute_mideye_eyed(self, annotations):
    reye_y, reye_x, leye_y, leye_x = self._get_eyes_coordinates(annotations)
    mideye_y = (reye_y + leye_y) / 2.
    mideye_x = (reye_x + leye_x) / 2.
    eye_d = self._distance(reye_y, reye_x, leye_y, leye_x)
    return (eye_d, mideye_y, mideye_x)

  def extract_keypoints(self, image, annotations_arg):
    """Executes the keypoints extractor"""

    # Check if there is a need to extract annotations
    annotations = None
    if self.m_fixed_annotations != None: annotations = self.m_fixed_annotations
    elif annotations_arg != None: annotations = annotations_arg

    if annotations != None:
      # Allocates array for keypoints
      keypoints = numpy.ndarray(shape=(len(annotations),2), dtype=numpy.float64)

      if self.m_crop_image: # If the original image is cropped
        c_eye_d, c_mideye_y, c_mideye_x = self._compute_mideye_eyed(self.m_cropped_positions)
        c_mideye = numpy.array([c_mideye_y, c_mideye_x], dtype=numpy.float64)
        if self.m_cropped_domain_annotations == True: # If annotations are in the cropped domain
          if self.m_relative_annotations: # If relative annotations, get the absolute coordinates of the keypoints
            c = 0
            for k in sorted(annotations):
              keypoints[c,:] = c_mideye + c_eye_d * numpy.array(annotations[k])
              c += 1
          else: # Otherwise, keypoints coordinates are already absolute
            c=0
            for k in sorted(annotations):
              keypoints[c,:] = numpy.array(annotations[k])
              c += 1
        else: # If annotations are in the original image
          o_eye_d, o_mideye_y, o_mideye_x = self._compute_mideye_eyed(annotations)
          o_mideye = numpy.array([o_mideye_y, o_mideye_x], dtype=numpy.float64)
          if self.m_relative_annotations: # If relative annotations, get the absolute coordinates of the keypoints
            c = 0
            for k in sorted(annotations):
              keypoints[c,:] = c_mideye + c_eye_d * numpy.array(annotations[k])
              c += 1
          else: # Otherwise, convert absolute annotations from the original domain to the cropped one
            c=0
            for k in sorted(annotations):
              keypoints[c,:] = c_mideye + c_eye_d * (numpy.array(annotations[k]) - o_mideye) / o_eye_d
              c += 1
      else: # cropped_domain does not matter since original and cropped domain are identical
        if self.m_relative_annotations:
          c_eye_d, c_mideye_y, c_mideye_x = self._compute_mideye_eyed(self.m_cropped_positions)
          c_mideye = numpy.array([c_mideye_y, c_mideye_x], dtype=numpy.float64)
          c = 0
          for k in sorted(annotations):
            keypoints[c,:] = c_mideye + c_eye_d * numpy.array(annotations[k])
            c += 1
        else:
          c=0
          for k in sorted(annotations):
            keypoints[c,:] = numpy.array(annotations[k])
            c += 1
    else:
      keypoints = numpy.ndarray(shape=(0,2), dtype=numpy.float64)

    # Returns the image (grayscale) and the keypoints
    return (image, keypoints)

  def __call__(self, image, annotations = None):
    """Reads the input image, and the annotations, and write them to an HDF5 file"""
    # crop the face using the base class method
    if self.m_crop_image == True:
      image2 = self.crop_face(image, annotations)
    else:
      # convert to grayscale
      image2 = utils.gray_channel(image, self.m_color_channel)

    return self.extract_keypoints(image2, annotations)

  def save_data(self, image, image_file):
    f = bob.io.base.HDF5File(image_file, 'w')
    f.set('image', image[0])
    f.set('annotations', image[1])

  def read_data(self, image_file):
    f = bob.io.base.HDF5File(image_file, 'r')
    image = f.read('image')
    annotations = f.read('annotations')
    return (image, annotations)
