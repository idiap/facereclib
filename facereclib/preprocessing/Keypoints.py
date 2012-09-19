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

import bob
import numpy
import math
from .. import utils

class Keypoints:
  """Extracts keypoints"""

  def __init__(self, config):
    self.m_config = config

  def extract_keypoints(self, image, annotations):
    """Executes the keypoints extractor"""
    # convert to grayscale
    # TODO: Use the utils... function instead
    image = bob.ip.rgb_to_gray(image)

    # Creates a keypoints numpy array
    # (storing keypoints using the alphabetical order of the labels)
    if annotations:
      keypoints = numpy.ndarray(shape=(len(annotations),2), dtype=numpy.float64)
      c=0
      for k in sorted(annotations):
        keypoints[c,:] = annotations[k]
        c=c+1
    else:
      keypoints = numpy.ndarray(shape=(0,2), dtype=numpy.float64)

    # Returns the image (grayscale) and the keypoints
    return (image, keypoints)

  def __call__(self, input_file, annotations = None):
    """Reads the input image, and the annotations, and write them to an HDF5 file"""
    return self.extract_keypoints(input_file, annotations)

  def save_image(self, image, image_file):
    f = bob.io.HDF5File(image_file, 'w')
    f.set('image', image[0])
    f.set('annotations', image[1])

  def read_image(self, image_file):
    f = bob.io.HDF5File(image_file, 'r')
    image = f.read('image')
    annotations = f.read('annotations')
    return (image, annotations)
