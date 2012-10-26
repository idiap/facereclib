#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Tue Oct  2 12:12:39 CEST 2012
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
import os

from .. import utils

class Preprocessor:
  """This is the base class for all preprocessors.
  It defines the minimum requirements for all derived proprocessor classes.
  """

  def __init__(self):
    # Each class needs to have a constructor taking
    # all the parameters that are required for the preprocessing as arguments
    pass

  # The call function (i.e. the operator() in C++ terms)
  def __call__(self, image, annotations):
    """This is the call function that you have to overwrite in the derived class.
    The parameters that this function will receive are:

     image
       The image that needs preprocessing

    annotations:
      The annotations (if any), as a dictionary from annotation type to the position.
      Usually, at least two annotations are given, namely: {'reye':(re_y,re_x), 'leye':(le_y,le_x)}
    """
    raise NotImplementedError("Please overwrite this function in your derived class")


  ############################################################
  ### Special functions that might be overwritten on need
  ############################################################

  def read_original_image(self, image_name):
    """Reads the *original* image from file.
    In this base class implementation, it uses bob.io.load to do that.
    If you have different format (e.g., not even images), please overwrite this function.
    """
    return bob.io.load(image_name)


  def save_image(self, image, image_file):
    """Saves the given *preprocessed* image to a file with the given name.
    In this base class implementation:

    - If the given image has a 'save' attribute, it calls image.save(bob.io.HDF5File(image_file), 'w').
      In this case, the given image_file might be either a file name or a bob.io.HDF5File.
    - Otherwise, it uses bob.io.save to do that.

    If you have a different format (e.g. not images), please overwrite this function.
    """
    utils.ensure_dir(os.path.dirname(image_file))
    if hasattr(image, 'save'):
      # this is some class that supports saving itself
      image.save(bob.io.HDF5File(image_file, "w"))
    else:
      bob.io.save(image, image_file)


  def read_image(self, image_name):
    """Reads the *preprocessed* image from file.
    In this base class implementation, it uses bob.io.load to do that.
    If you have different format (e.g., not even images), please overwrite this function.
    """
    return bob.io.load(image_name)

