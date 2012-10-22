#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

"""Image preprocessing tools"""

import bob
import numpy
import math
from .. import utils

from Preprocessor import Preprocessor
from FaceCrop import FaceCrop
from TanTriggs import TanTriggs, TanTriggsVideo
from HistogramEqualization import HistogramEqualization
from SelfQuotientImage import SelfQuotientImage
from INormLBP import INormLBP
from Keypoints import Keypoints
from Cepstral import Cepstral


class NullPreprocessor (Preprocessor):
  """Skips proprocessing files by simply copying the contents into an hdf5 file
  (and perform gray scale conversion if required)"""
  def __init__(self, color_channel = 'gray'):
    Preprocessor.__init__(self)
    self.m_color_channel = color_channel

  def __call__(self, image, annotations = None):
    # convert to grayscale
    image = utils.gray_channel(image, self.m_color_channel)
    return image.astype(numpy.float64)


