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

import numpy

def sparsify(array):
  """This function generates a sparse histogram from a non-sparse one."""
  if len(array.shape) == 2 and array.shape[0] == 2:
    return array
  assert len(array.shape) == 1
  indices = []
  values = []
  for i in range(array.shape[0]):
    if array[i] != 0.:
      indices.append(i)
      values.append(array[i])

  return numpy.array([indices, values], dtype = numpy.float64)
