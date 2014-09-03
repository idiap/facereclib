#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.learn.linear
import bob.io.base

import numpy
import scipy.spatial

from .Tool import Tool
from .. import utils

class PCA (Tool):
  """Tool for computing eigenfaces"""

  def __init__(
      self,
      subspace_dimension,  # if int, number of subspace dimensions; if float, percentage of variance to keep
      distance_function = scipy.spatial.distance.euclidean,
      is_distance_function = True,
      uses_variances = False,
      **kwargs  # parameters directly sent to the base class
  ):

    """Initializes the PCA tool with the given setup"""
    # call base class constructor and register that the tool performs a projection
    Tool.__init__(
        self,
        performs_projection = True,

        subspace_dimension = subspace_dimension,
        distance_function = str(distance_function),
        is_distance_function = is_distance_function,
        uses_variances = uses_variances,

        **kwargs
    )

    self.m_subspace_dim = subspace_dimension
    self.m_machine = None
    self.m_distance_function = distance_function
    self.m_factor = -1. if is_distance_function else 1.
    self.m_uses_variances = uses_variances


  def train_projector(self, training_features, projector_file):
    """Generates the PCA covariance matrix"""
    # Initializes the data
    data = numpy.vstack([feature.flatten() for feature in training_features])

    utils.info("  -> Training LinearMachine using PCA")
    t = bob.learn.linear.PCATrainer()
    self.m_machine, self.m_variances = t.train(data)
    # For re-shaping, we need to copy...
    self.m_variances = self.m_variances.copy()

    # compute variance percentage, if desired
    if isinstance(self.m_subspace_dim, float):
      cummulated = numpy.cumsum(self.m_variances) / numpy.sum(self.m_variances)
      for index in range(len(cummulated)):
        if cummulated[index] > self.m_subspace_dim:
          self.m_subspace_dim = index
          break
      self.m_subspace_dim = index

    utils.info("    ... Keeping %d PCA dimensions" % self.m_subspace_dim)

    # re-shape machine
    self.m_machine.resize(self.m_machine.shape[0], self.m_subspace_dim)
    self.m_variances.resize(self.m_subspace_dim)

    f = bob.io.base.HDF5File(projector_file, "w")
    f.set("Eigenvalues", self.m_variances)
    f.create_group("Machine")
    f.cd("/Machine")
    self.m_machine.save(f)


  def load_projector(self, projector_file):
    """Reads the PCA projection matrix from file"""
    # read PCA projector
    f = bob.io.base.HDF5File(projector_file)
    self.m_variances = f.read("Eigenvalues")
    f.cd("/Machine")
    self.m_machine = bob.learn.linear.Machine(f)
    # Allocates an array for the projected data
    self.m_projected_feature = numpy.ndarray(self.m_machine.shape[1], numpy.float64)

  def project(self, feature):
    """Projects the data using the stored covariance matrix"""
    # Projects the data
    self.m_machine(feature, self.m_projected_feature)
    # return the projected data
    return self.m_projected_feature

  def enroll(self, enroll_features):
    """Enrolls the model by computing an average of the given input vectors"""
    assert len(enroll_features)
    # just store all the features
    model = numpy.zeros((len(enroll_features), enroll_features[0].shape[0]), numpy.float64)
    for n, feature in enumerate(enroll_features):
      model[n,:] += feature[:]

    # return enrolled model
    return model


  def score(self, model, probe):
    """Computes the distance of the model to the probe using the distance function taken from the config file"""
    # return the negative distance (as a similarity measure)
    if len(model.shape) == 2:
      # we have multiple models, so we use the multiple model scoring
      return self.score_for_multiple_models(model, probe)
    elif self.m_uses_variances:
      # single model, single probe (multiple probes have already been handled)
      return self.m_factor * self.m_distance_function(model, probe, self.m_variances)
    else:
      # single model, single probe (multiple probes have already been handled)
      return self.m_factor * self.m_distance_function(model, probe)
