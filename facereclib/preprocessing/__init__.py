#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

"""Image preprocessing tools"""

import bob
import numpy
import math
from .. import utils

class NullPreprocessor:
  """Skips proprocessing files by simply copying the contents into an hdf5 file 
  (and perform gray scale conversion if required)"""
  def __init__(self, config):
    self.m_color_channel = config.color_channel if hasattr(config, 'color_channel') else 'gray'
    
  def __call__(self, input_file, output_file, annotations = None):
    image = bob.io.load(str(input_file))
    # convert to grayscale
    image = utils.gray_channel(image, self.m_color_channel)
    image = image.astype(numpy.float64)
    bob.io.save(image, output_file)

from FaceCrop import FaceCrop, StaticFaceCrop
from TanTriggs import TanTriggs, StaticTanTriggs, TanTriggsVideo
from HistogramEqualization import HistogramEqualization
from SelfQuotientImage import SelfQuotientImage
from INormLBP import INormLBP
