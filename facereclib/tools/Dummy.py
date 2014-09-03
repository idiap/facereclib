#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Fri Oct 26 17:05:40 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from .Tool import Tool

import numpy
import scipy.spatial
from .. import utils


class Dummy (Tool):
  """This class is used to test all the possible functions of the tool chain, but it does basically nothing."""

  def __init__(self):
    """Generates a test value that is read and written"""

    # call base class constructor registering that this tool performs everything.
    Tool.__init__(
        self,
        performs_projection = True,
        use_projected_features_for_enrollment = True,
        requires_enroller_training = True
    )

    self.m_test_value = numpy.array([[1,2,3], [4,5,6], [7,8,9]], dtype = numpy.uint8)

  def __test__(self, file_name):
    """Simply tests that the read data is consistent"""
    test_value = utils.load(file_name)
    for y in range(3):
      for x in range(3):
        assert test_value[y,x] == self.m_test_value[y,x]

  def train_projector(self, train_files, projector_file):
    """Does not train the projector, but writes some file"""
    utils.debug("DummyTool: Training projector %s with %d training files" % (projector_file, len(train_files)))
    # save something
    utils.save(self.m_test_value, projector_file)

  def load_projector(self, projector_file):
    """Loads the test value from file and compares it with the desired one"""
    utils.debug("DummyTool: Loading projector file %s" % projector_file)
    self.__test__(projector_file)

  def project(self, feature):
    """Just returns the feature since this dummy implementation does not really project the data"""
    return feature

  def train_enroller(self, train_files, enroller_file):
    """Does not train the projector, but writes some file"""
    utils.debug("DummyTool: Training enroller %s using %d features" % (enroller_file, len(train_files)))
    # save something
    utils.save(self.m_test_value, enroller_file)

  def load_enroller(self, enroller_file):
    """Loads the test value from file and compares it with the desired one"""
    utils.debug("DummyTool: Loading enroller %s" % enroller_file)
    self.__test__(enroller_file)

  def enroll(self, enroll_features):
    """Returns the first feature as the model"""
    utils.debug("DummyTool: Enrolling model using %d features" % len(enroll_features))
    assert len(enroll_features)
    # just return the first feature
    return enroll_features[0]

  def save_feature(self, feature, feature_file):
    """Saves the given feature to the given file"""
    utils.debug("DummyTool: Saving feature of length %d to file %s" % (feature.shape[0], feature_file))
    utils.save(feature, feature_file)

  def read_feature(self, feature_file):
    """Reads the feature from the given file"""
    utils.debug("DummyTool: Reading feature from file %s" % feature_file)
    return utils.load(feature_file)

  def save_model(self, model, model_file):
    """Writes the model to the given model file"""
    utils.debug("DummyTool: Saving model of length %d to file %s" % (model.shape[0], model_file))
    utils.save(model, model_file)

  def read_model(self, model_file):
    """Reads the model from file"""
    utils.debug("DummyTool: Reading model from file %s" % model_file)
    return utils.load(model_file)

  def read_probe(self, probe_file):
    """Reads the probe from file"""
    utils.debug("DummyTool: Reading probe from file %s" % probe_file)
    return utils.load(probe_file)

  def score(self, model, probe):
    """Returns the Euclidean distance between model and probe"""
    return scipy.spatial.distance.euclidean(model, probe)

