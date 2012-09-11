#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy


class PCATool:
  """Tool for computing eigenfaces"""

  def __init__(self, setup):
    """Initializes the PCA tool with the given setup"""
    self.m_subspace_dim = setup.SUBSPACE_DIMENSIONS
    self.m_machine = None
    self.m_distance_function = setup.distance_function
    self.m_factor = -1 if not hasattr(setup, 'IS_DISTANCE_FUNCTION') or setup.IS_DISTANCE_FUNCTION else 1.


  def train_projector(self, training_features, projector_file):
    """Generates the PCA covariance matrix"""
    # Initializes an arrayset for the data
    data = bob.io.Arrayset()
    for k in sorted(training_features.keys()):
      # Loads the file
      feature = bob.io.load(str(training_features[k]))
      # Appends in the arrayset
      data.append(feature)

    print "Training LinearMachine using PCA (SVD)"
    t = bob.trainer.SVDPCATrainer()
    self.m_machine, __eig_vals = t.train(data)
    # Machine: get shape, then resize
    self.m_machine.resize(self.m_machine.shape[0], self.m_subspace_dim)
    self.m_machine.save(bob.io.HDF5File(projector_file, "w"))


  def load_projector(self, projector_file):
    """Reads the PCA projection matrix from file"""
    # read PCA projector
    self.m_machine = bob.machine.LinearMachine(bob.io.HDF5File(projector_file))
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
    model = None
    for feature in enroll_features:
      if model == None:
        model = numpy.zeros(feature.shape, numpy.float64)

      model[:] += feature[:]

    # Normalizes the model
    model /= float(len(enroll_features))

    # return enrollled model
    return model


  def score(self, model, probe):
    """Computes the distance of the model to the probe using the distance function taken from the config file"""
    # return the negative distance (as a similarity measure)
    return self.m_factor * self.m_distance_function(model, probe)

