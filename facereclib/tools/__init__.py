#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

"""Tool chain for computing verification scores"""

from Tool import Tool
from Dummy import Dummy
from GaborJets import GaborJets
from LGBPHS import LGBPHS
from UBMGMM import UBMGMM, UBMGMMRegular, UBMGMMVideo
from JFA import JFA
from ISV import ISV, ISVVideo
from IVector import IVector
from PCA import PCA
from LDA import LDA
from PLDA import PLDA
from BIC import BIC
from ParallelUBMGMM import ParallelUBMGMM

