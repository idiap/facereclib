#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy


class LDATool:
  """Tool for computing linear discriminant analysis (socalled Fisher faces)"""

  def __init__(self, setup):
    """Initializes the LDA tool with the given configuration"""
    self.m_config = setup
    self.m_pca_subpace_size = setup.pca_subspace if hasattr(setup, 'pca_subspace') else None
    self.m_lda_subspace_size = setup.lda_subspace if hasattr(setup, 'lda_subspace') else None
    self.m_machine = None
    self.m_distance_function = setup.distance_function
    self.m_factor = -1 if not hasattr(setup, 'is_distance_function') or setup.is_distance_function else 1.
    
    # declare that this LDA tool requires training data separated by identity 
    self.use_training_features_sorted_by_identity = True
    
  def __read_data__(self, training_files):
    data = []
    for client in sorted(training_files.keys()):
      print client
      # Arrayset for this client
      client_data = bob.io.Arrayset()
      client_files = training_files[client]
      # at least two files per client are required!
      if len(client_files) < 2:
        print "Skipping", len(client_files)
        continue
      for k in sorted(client_files.keys()):
        # Loads the file
        feature = bob.io.load( str(client_files[k]) )
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

    print "Training LinearMachine using PCA (SVD)"
    t = bob.trainer.SVDPCATrainer()
    machine, __eig_vals = t.train(data)
    # limit number of pcs
    machine.resize(machine.shape[0], self.m_pca_subpace_size)
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
    

  def train_projector(self, training_files, projector_file):
    """Generates the LDA projection matrix from the given features (that are sorted by identity)"""
    # Initializes an arrayset for the data
    data = self.__read_data__(training_files)
    
    if self.m_pca_subpace_size:
      pca_machine = self.__train_pca__(data)
      data = self.__perform_pca__(pca_machine, data)

    print "Training LinearMachine using LDA"
    t = bob.trainer.FisherLDATrainer()
    self.m_machine, __eig_vals = t.train(data)
    
    if self.m_pca_subpace_size:
      # compute combined PCA/LDA projection matrix
      combined_matrix = numpy.dot(pca_machine.weights, self.m_machine.weights)
      # set new weigth matrix (and new mean vector) of novel machine
      self.m_machine = bob.machine.LinearMachine(combined_matrix)
      self.m_machine.input_subtract = pca_machine.input_subtract

    # resize the machine if desired
    if self.m_lda_subspace_size:
      self.m_machine.resize(self.m_machine.shape[0], self.m_lda_subspace_size)
    
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
    
  def enrol(self, enrol_features):
    """Enrols the model by computing an average of the given input vectors"""
    model = None
    for feature in enrol_features:
      if model == None:
        model = numpy.zeros(feature.shape, numpy.float64)

      model[:] += feature[:]
        
    # Normalizes the model
    model /= float(len(enrol_features))

    # return enroled model    
    return model

    
  def score(self, model, probe):
    """Computes the distance of the model to the probe using the distance function taken from the config file"""
    # return the negative distance (as a similarity measure)
    return self.m_factor * self.m_distance_function(model, probe)

    

