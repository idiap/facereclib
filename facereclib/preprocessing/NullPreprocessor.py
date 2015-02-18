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

from .Preprocessor import Preprocessor
from .. import utils
import numpy

class NullPreprocessor (Preprocessor):
  """Skips proprocessing by simply copying the file contents into an hdf5 file
  (and perform gray scale conversion if required)"""

  def __init__(self, color_channel = 'gray', **kwargs):
    Preprocessor.__init__(self, color_channel=color_channel, **kwargs)
    self.m_color_channel = color_channel

  def __call__(self, data, annotations = None):
    """Just perform gray scale conversion, ignore the annotations."""
    # convert to grayscale
    image = utils.gray_channel(data, self.m_color_channel)
    return image.astype(numpy.float64)
