#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu Jul 19 17:09:55 CEST 2012
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

from .resources import resource_keys, load_resource
import numpy

def configuration_file(name, resource, dir = None, module = 'facereclib.configurations'):
  """Reads configuration file or the registered resource with the given name."""
  # test if the resource is known
  if name in resource_keys(resource):
    # resource registered, just load it
    return load_resource(name, resource)
  else: # resource not registered, but available...
    # import resource (actually this is a hack, but better than dealing with file names...)
    exec "from " + module + "." + dir + " import " + name
    # get the database defined in the resource
    return eval(name + "." + resource)

def random_training_set(shape, count, minimum = 0, maximum = 1):
  """Returns a random training set with the given shape and the given number of elements."""
  # generate a random sequence of features
  numpy.random.seed(42)
  return [numpy.random.random(shape) * (maximum - minimum) + minimum for i in range(count)]

def random_training_set_by_id(shape, count = 50, minimum = 0, maximum = 1):
  # generate a random sequence of features
  numpy.random.seed(42)
  train_set = []
  for i in range(count):
    train_set.append([numpy.random.random(shape) * (maximum - minimum) + minimum for j in range(count)])
  return train_set

