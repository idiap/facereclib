#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""Features for face recognition"""

from Extractor import Extractor
from DCT import DCTBlocks, DCTBlocksVideo
from LGBPHS import LGBPHS
from GridGraph import GridGraph
from Eigenface import Eigenface
from SIFTKeypoints import SIFTKeypoints


import numpy

class Linearize (Extractor):
  """Extracts pixel-based features by simply concatenating all pixels of the image into one long vector"""

  def __init__(self):
    """Nothing to be done here."""
    Extractor.__init__(self)

  def __call__(self, image):
    """Takes an image of arbitrary dimensions and linearizes it into a 1D vector"""
    return numpy.reshape(image, image.size)

