#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu Oct 25 10:05:55 CEST 2012
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

import imp
import os
import pkg_resources


def read_config_file(file, keyword = None):
  """Use this function to read the given configuration file.
  If a keyword is specified, only the configuration according to this keyword is returned.
  Otherwise a dictionary of the configurations read from the configuration file is returned."""

  if not os.path.exists(file):
    raise IOError("The given configuration file '%s' could not be found" % file)

  import string
  import random
  tmp_config = "".join(random.sample(string.letters, 10))
  config = imp.load_source(tmp_config, file)

  if not keyword:
    return config

  if not hasattr(config, keyword):
    raise ImportError("The desired keyword '%s' does not exist in your configuration file '%s'." %(keyword, file))

  return eval('config.' + keyword)


def read_resource(resource, keyword):

  # first, look if the resource is a file name
  if os.path.exists(resource):
    return read_config_file(resource, keyword)

  # now, we ckeck if the resource is registered as an entry point in the resource files
  for entry_point in pkg_resources.iter_entry_points('facereclib.' + keyword):
    file = entry_point.load()
    print file

