#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy

from .Tool import Tool
from . import UBMGMMTool
from .. import utils

class UBMGMMVideoTool(UBMGMMTool):
  """Tool chain for computing Universal Background Models and Gaussian Mixture Models of the features"""


  def __init__(self, setup):
    UBMGMMTool.__init__(self, setup)


  def train_projector(self, train_features, projector_file):
    """Computes the Universal Background Model from the training ("world") data"""

    utils.info("  -> Training UBM model with %d training files" % len(train_files))

    # Loads the data into an array
    data_list = []
    for frame_container in train_files:
      for data in self.m_config.frame_selector_for_projector_training(frame_container):
        data_list.append(data)
    array = numpy.vstack(data_list)

    self._train_projector_using_array(array, projector_file)


  def read_feature(self, feature_file):
    return utils.video.FrameContainer(str(feature_file))


  def project(self, frame_container, frame_selector = None):
    """Computes GMM statistics against a UBM, given an input video.FrameContainer"""

    if frame_selector is None:
      frame_selector = self.m_config.frame_selector_for_projection

    # Collect all feature vectors across all frames in a single array set
    array = numpy.vstack([data for data in frame_selector(frame_container)])
    return self._project_using_array(array)


  def enroll(self, frame_containers):
    """Enrolls a GMM using MAP adaptation, given a list of video.FrameContainers"""

    # Load the data into an array
    data_list = []
    for frame_container in frame_containers:
      for data in self.m_config.frame_selector_for_enroll(frame_container):
        data_list.append(data)
    array = numpy.vstack(data_list)

    # Use the array to train a GMM and return it
    return self._enroll_using_array(array)

