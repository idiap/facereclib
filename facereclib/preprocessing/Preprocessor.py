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

import bob.io.base
import bob.io.image

import os

from .. import utils

class Preprocessor:
  """This is the base class for all preprocessors.
  It defines the minimum requirements for all derived proprocessor classes.
  """

  def __init__(self, **kwargs):
    # Each class needs to have a constructor taking
    # all the parameters that are required for the preprocessing as arguments
    self._kwargs = kwargs
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


  def __str__(self):
    """This function returns a string containing all parameters of this class (and its derived class)."""
    return "%s(%s)" % (str(self.__class__), ", ".join(["%s=%s" % (key, value) for key,value in self._kwargs.items() if value is not None]))

  ############################################################
  ### Special functions that might be overwritten on need
  ############################################################

  def read_original_data(self, original_file_name):
    """Reads the *original* data (usually an image) from file.
    In this base class implementation, it uses ``bob.io.base.load`` to do that.
    If you have different format, please overwrite this function.
    """
    return bob.io.base.load(original_file_name)


  def save_data(self, data, data_file):
    """Saves the given *preprocessed* data to a file with the given name.
    In this base class implementation:

    - If the given data has a ``save`` attribute, it calls ``data.save(bob.io.base.HDF5File(data_file), 'w')``.
      In this case, the given data_file might be either a file name or a bob.io.base.HDF5File.
    - Otherwise, it uses ``bob.io.base.save`` to do that.

    If you have a different format (e.g. not images), please overwrite this function.
    """
    utils.save(data, data_file)


  def read_data(self, data_file):
    """Reads the *preprocessed* data from file.
    In this base class implementation, it uses ``bob.io.base.load`` to do that.
    If you have different format, please overwrite this function.
    """
    return utils.load(data_file)


  def quality(self):
    """Returns the quality of the last preprocessed image.
    This quality term is application dependent.
    By default, None is returned."""
    return None
