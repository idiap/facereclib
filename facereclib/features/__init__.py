#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""Features for face recognition"""

from DCT import DCTBlocks
from DCTVideo import DCTBlocksVideo
from LGBPHS import LGBPHS
from GridGraph import GridGraph
from Eigenface import Eigenface
from SIFTKeypoints import SIFTKeypoints


import numpy

class Linearize:
  """Extracts Eigenface feature by simply concatenating all pixels of the image into one long vector"""
  
  def __init__(self, setup):
    """Nothing to be done here."""
    pass
    
  def __call__(self, image):
    """Takes a 2D image and linearizes it into a 1D vector"""
    return numpy.reshape(image, image.size)

