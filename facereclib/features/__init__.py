#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""Features for face recognition"""

from .Extractor import Extractor
from .Linearize import Linearize
from .DCT import DCTBlocks
from .LGBPHS import LGBPHS
from .GridGraph import GridGraph
from .Eigenface import Eigenface
from .SIFTKeypoints import SIFTKeypoints
from .SIFTBobKeypoints import SIFTBobKeypoints

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
