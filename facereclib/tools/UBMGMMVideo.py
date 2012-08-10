#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy
from . import UBMGMMTool
from .. import utils

class UBMGMMVideoTool(UBMGMMTool):
  """Tool chain for computing Universal Background Models and Gaussian Mixture Models of the features"""
  

  def __init__(self, setup):
    UBMGMMTool.__init__(self, setup)


  def train_projector(self, train_files, projector_file):
    """Computes the Universal Background Model from the training ("world") data"""

    print "Training UBM model with %d training files" % len(train_files)
    
    # Loads the data into an Arrayset
    arrayset = bob.io.Arrayset()
    for k in sorted(train_files.keys()):
      print "-> UBMGMMVideoTool.train_projector processing train_file: " + str(train_files[k]) # TODO: remove debug
      frame_container = utils.VideoFrameContainer(str(train_files[k]))
      for data in self.m_config.frame_selector_for_train_projector(frame_container):
        arrayset.extend(data)
    print "-> UBMGMMVideoTool.train_projector ready to train using " + str(len(arrayset)) + " vectors" # TODO: remove debug

    self._train_projector_using_arrayset(arrayset, projector_file)
   

  def read_feature(self, feature_file):
    return utils.VideoFrameContainer(str(feature_file))


  def project(self, frame_container, frame_selector = None):
    """Computes GMM statistics against a UBM, given an input VideoFrameContainer"""
    
    if frame_selector is None:
      frame_selector = self.m_config.frame_selector_for_project

    # Collect all feature vectors across all frames in a single array set
    arrayset = bob.io.Arrayset()
    for data in frame_selector(frame_container):
      arrayset.extend(data)
    print "-> UBMGMMVideoTool.project ready to project using " + str(len(arrayset)) + " vectors" # TODO: remove debug
    return self._project_using_arrayset(arrayset)
    

  def enrol(self, frame_containers):
    """Enrols a GMM using MAP adaptation, given a list of VideoFrameContainers"""
    
    # Load the data into an Arrayset
    arrayset = bob.io.Arrayset()
    for frame_container in frame_containers:
      for data in self.m_config.frame_selector_for_enrol(frame_container):
        arrayset.extend(data)
    print "-> UBMGMMVideoTool.enrol ready to enrol using " + str(len(arrayset)) + " vectors" # TODO: remove debug

    # Use the Arrayset to train a GMM and return it
    return self._enrol_using_arrayset(arrayset)

