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
from .logger import warn


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


def _get_entry_points(keyword):
  return  [entry_point for entry_point in pkg_resources.iter_entry_points('facereclib.' + keyword)]

def resource_keys(keyword):
  """Reads and returns all resources that are registered with the given keyword."""
  return sorted([entry_point.name for entry_point in _get_entry_points(keyword)])

def load_resource(resource, keyword, imports = []):
  """Loads the given resource that is registered with the given keyword.
  The resource can be:

    * a resource as defined in the setup.py
    * a configuration file
    * a string defining the construction of an object. If imports are required for the construction of this object, they can be given as list of strings.

  In any case, the resulting resource object is returned.
  """

  # first, look if the resource is a file name
  if os.path.exists(resource):
    return read_config_file(resource, keyword)

  # now, we ckeck if the resource is registered as an entry point in the resource files
  entry_points = [entry_point for entry_point in _get_entry_points(keyword) if entry_point.name == resource]

  if len(entry_points):
    if len(entry_points) == 1:
      return entry_points[0].load()
    else:
      # TODO: extract current package name and use this one, if possible

      # Now: check if there are only two entry points, and one is from the facereclib, then use the other one
      index = -1
      if len(entry_points) == 2:
        print entry_points[0].dist.project_name
        if entry_points[0].dist.project_name == 'facereclib': index = 1
        elif entry_points[1].dist.project_name == 'facereclib': index = 0

      if index != -1:
        warn("RESOURCES: Using the resource '%s' from '%s', and ignoring the one from '%s'" %(resource, entry_points[index].module_name, entry_points[1-index].module_name))
        return entry_points[index].load()
      else:
        raise ImportError("Under the desired name '%s', there are multiple entry points defined: %s" %(resource, [entry_point.module_name for entry_point in entry_points]))

  # if the resource is neither a config file nor an entry point,
  # just execute it as a command

  try:
    # first, execute all import commands that are required
    for i in imports:
      exec "import %s"%i
    # now, evaluate the resources
    return eval(resource)

  except Exception as e:
    raise ImportError("The given command line option '%s' is neither a resource for a '%s', nor an existing configuration file, nor could be interpreted as a command (error: %s)"%(resource, keyword, str(e)))


def print_resources(keyword):
  """Prints a detailed list of resources that are registered with the given keyword."""
  entry_points = _get_entry_points(keyword)
  for entry_point in entry_points:
    print "-", entry_point.name, "(" + str(entry_point.dist) + ")  -->", entry_point.module_name, ":", entry_point.attrs[0]

def print_all_resources():
  """Prints a detailed list of all resources that are registered."""
  print "List of registered databases:"
  print_resources('database')
  print
  print "List of registered preprocessors:"
  print_resources('preprocessor')
  print
  print "List of registered feature extractors:"
  print_resources('feature_extractor')
  print
  print "List of registered recognition algorithms:"
  print_resources('tool')
  print
