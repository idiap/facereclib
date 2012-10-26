#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

"""Tool chain for computing verification scores"""

from Tool import Tool
from Dummy import DummyTool
from GaborJets import GaborJetTool
from LGBPHS import LGBPHSTool
from UBMGMM import UBMGMMTool, UBMGMMRegularTool, UBMGMMVideoTool
from JFA import JFATool
from ISV import ISVTool, ISVVideoTool
from PCA import PCATool
from LDA import LDATool
from PLDA import PLDATool
from BIC import BICTool

