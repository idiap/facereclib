#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Fri Oct 26 17:05:40 CEST 2012
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

from .Extractor import Extractor
import numpy

class Linearize (Extractor):
  """Extracts pixel-based features by simply concatenating all pixels of the image into one long vector"""

  def __init__(self):
    """Nothing to be done here."""
    Extractor.__init__(self)

  def __call__(self, image):
    """Takes an image of arbitrary dimensions and linearizes it into a 1D vector"""
    return numpy.reshape(image, image.size)
