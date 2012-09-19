#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy

from .. import utils

class LDATool:
  """Tool for computing linear discriminant analysis (so-called Fisher faces)"""

  def __init__(self, setup):
    """Initializes the LDA tool with the given configuration"""
    self.m_pca_subspace = setup.PCA_SUBSPACE_DIMENSION if hasattr(setup, 'PCA_SUBSPACE_DIMENSION') else None
    self.m_lda_subspace = setup.LDA_SUBSPACE_DIMENSION if hasattr(setup, 'LDA_SUBSPACE_DIMENSION') else None
    if self.m_pca_subspace and self.m_lda_subspace and self.m_pca_subspace < self.m_lda_subspace:
      raise ValueError("The LDA subspace is larger than the PCA subspace size. This won't work properly. Please check your setup!")
    self.m_machine = None
    self.m_distance_function = setup.distance_function
    self.m_factor = -1 if not hasattr(setup, 'is_distance_function') or setup.is_distance_function else 1.

    # declare that this LDA tool requires training data separated by identity
    self.use_training_features_sorted_by_identity = True

  def __read_data__(self, training_files):
    data = []
    for client in sorted(training_files.keys()):
      # Arrayset for this client
      client_data = bob.io.Arrayset()
      client_files = training_files[client]
      # at least two files per client are required!
      if len(client_files) < 2:
        utils.warn("Skipping client with id %s since the number of client files is only %d" %(client, len(client_files)))
        continue
      for k in sorted(client_files.keys()):
        # Loads the file
        feature = client_files[k]
        # Appends in the arrayset; assure that the data is 1-dimensional
        client_data.append(numpy.reshape(feature, feature.size))
      data.append(client_data)

    # Returns the list of Arraysets
    return data

  def __train_pca__(self, training_set):
    """Trains and returns a LinearMachine that is trained using PCA"""
    data = bob.io.Arrayset()
    for client in training_set:
      for feature in client:
        # Appends in the arrayset
        data.append(feature)

    utils.info("  -> Training LinearMachine using PCA (SVD)")
    t = bob.trainer.SVDPCATrainer()
    machine, __eig_vals = t.train(data)
    # limit number of pcs
    machine.resize(machine.shape[0], self.m_pca_subspace)
    return machine


  def __perform_pca__(self, machine, training_set):
    """Perform PCA on data"""
    data = []
    for client in training_set:
      client_data = bob.io.Arrayset()
      for feature in client:
        # project data
        projected_feature = numpy.ndarray(machine.shape[1], numpy.float64)
        machine(feature, projected_feature)
        # overwrite data in training set
        client_data.append(projected_feature)
      data.append(client_data)
    return data


  def train_projector(self, training_features, projector_file):
    """Generates the LDA projection matrix from the given features (that are sorted by identity)"""
    # Initializes an arrayset for the data
    data = self.__read_data__(training_features)

    if self.m_pca_subspace:
      pca_machine = self.__train_pca__(data)
      data = self.__perform_pca__(pca_machine, data)

    utils.info("  -> Training LinearMachine using LDA")
    t = bob.trainer.FisherLDATrainer()
    self.m_machine, __eig_vals = t.train(data)

    if self.m_pca_subspace:
      # compute combined PCA/LDA projection matrix
      combined_matrix = numpy.dot(pca_machine.weights, self.m_machine.weights)
      # set new weigth matrix (and new mean vector) of novel machine
      self.m_machine = bob.machine.LinearMachine(combined_matrix)
      self.m_machine.input_subtract = pca_machine.input_subtract

    # resize the machine if desired
    if self.m_lda_subspace:
      self.m_machine.resize(self.m_machine.shape[0], self.m_lda_subspace)

    self.m_machine.save(bob.io.HDF5File(projector_file, "w"))


  def load_projector(self, projector_file):
    """Reads the UBM model from file"""
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

    # return enrolled model
    return model


  def score(self, model, probe):
    """Computes the distance of the model to the probe using the distance function taken from the config file"""
    # return the negative distance (as a similarity measure)
    return self.m_factor * self.m_distance_function(model, probe)



