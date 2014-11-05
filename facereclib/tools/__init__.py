#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

"""Tool chain for computing verification scores"""

from .Tool import Tool
from .Dummy import Dummy
from .GaborJets import GaborJets
from .LGBPHS import LGBPHS
from .UBMGMM import UBMGMM, UBMGMMRegular
from .JFA import JFA
from .ISV import ISV
from .IVector import IVector
from .PCA import PCA
from .LDA import LDA
from .PLDA import PLDA
from .BIC import BIC

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
