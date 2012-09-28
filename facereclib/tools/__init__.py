#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

"""Tool chain for computing verification scores"""

from GaborJets import GaborJetTool
from LGBPHS import LGBPHSTool
from UBMGMM import UBMGMMTool
from UBMGMMRegular import UBMGMMRegularTool
from UBMGMMVideo import UBMGMMVideoTool
from JFA import JFATool
from ISV import ISVTool
from ISVVideo import ISVVideoTool
from PCA import PCATool
from LDA import LDATool
from PLDA import PLDATool
from BIC import BICTool


import numpy
import bob
from .. import utils


class DummyTool:
  """This class is used to test all the possible functions of the tool chain, but it does basically nothing."""

  def __init__(self, setup):
    """Generates a test value that is read and written"""
    self.m_test_value = numpy.array([[1,2,3], [4,5,6], [7,8,9]], dtype = numpy.uint8)

  def __test__(self, file_name):
    """Simply tests that the read data is consistent"""
    test_value = bob.io.load(str(file_name))
    for y in range(3):
      for x in range(3):
        assert test_value[y,x] == self.m_test_value[y,x]

  def train_projector(self, train_files, projector_file):
    """Does not train the projector, but writes some file"""
    utils.debug("DummyTool: Training projector %s with %d training files" % (projector_file, len(train_files)))
    # save something
    bob.io.save(self.m_test_value, projector_file)

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
    bob.io.save(self.m_test_value, enroller_file)

  def load_enroller(self, enroller_file):
    """Loads the test value from file and compares it with the desired one"""
    utils.debug("DummyTool: Training enroller %s using %d features" % (enroller_file, len(train_files)))
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
    bob.io.save(feature, feature_file)

  def read_feature(self, feature_file):
    """Reads the feature from the given file"""
    utils.debug("DummyTool: Reading feature from file %s" % feature_file)
    return bob.io.load(feature_file)

  def save_model(self, model, model_file):
    """Writes the model to the given model file"""
    utils.debug("DummyTool: Saving model of length %d to file %s" % (model.shape[0], model_file))
    bob.io.save(model, model_file)

  def read_model(self, model_file):
    """Reads the model from file"""
    utils.debug("DummyTool: Reading model from file %s" % model_file)
    return bob.io.load(model_file)

  def read_probe(self, probe_file):
    """Reads the probe from file"""
    utils.debug("DummyTool: Reading probe from file %s" % probe_file)
    return bob.io.load(probe_file)

  def score(self, model, probe):
    """Returns the Euclidean distance between model and probe"""
    return bob.math.euclidean_distance(model, probe)

