#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy
from UBMGMM import UBMGMMTool
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
      frame_container = utils.VideoFrameContainer(str(train_files[k]))
      for data in frame_container.frames_data():
        arrayset.extend(data)

    self._train_projector_using_arrayset(arrayset, projector_file)
   

  def read_feature(self, feature_file):
    return utils.VideoFrameContainer(str(feature_file))


  def project(self, frame_container):
    """Computes GMM statistics against a UBM, given an input VideoFrameContainer"""
    
    # Collect all feature vectors across all frames in a single array set
    arrayset = bob.io.Arrayset()
    for data in frame_container.frames_data():
      arrayset.extend(data)
    return self._project_using_arrayset(arrayset)
    

  def enrol(self, frame_containers):
    """Enrols a GMM using MAP adaptation, given a list of VideoFrameContainers"""
    
    # Load the data into an Arrayset
    arrayset = bob.io.Arrayset()
    for frame_container in frame_containers:
      for data in frame_container.frames_data():
        arrayset.extend(data)

    # Use the Arrayset to train a GMM and return it
    return self._enrol_using_arrayset(arrayset)

