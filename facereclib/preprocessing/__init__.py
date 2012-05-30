#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

"""Image preprocessing tools"""

import bob
import numpy
import math

class NullPreprocessor:
  """Skips proprocessing files by simply copying the contents into an hdf5 file 
  (and perform gray scale conversion if required)"""
  def __init__(self, config):
    pass
    
  def __call__(self, input_file, output_file, eye_pos = None):
    image = bob.io.load(str(input_file))
    # convert to grayscale
    if(image.ndim == 3):
      image = bob.ip.rgb_to_gray(image)
    image = image.astype(numpy.float64)
    bob.io.save(image, output_file)

from FaceCrop import FaceCrop, StaticFaceCrop
from TanTriggs import TanTriggs, StaticTanTriggs, TanTriggsVideo

