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
    self.m_machine = None
    self.m_distance_function = self.m_config.distance_function
    
    # declare that this LDA tool requires training data separated by identity 
    self.use_training_features_sorted_by_identity = True
    
  def __read_data__(self, training_files):
    data = []
    for client in sorted(training_files.keys()):
      # Arrayset for this client
      client_data = bob.io.Arrayset()
      client_files = training_files[client]
      # at least two files per client are required!
      assert len(client_files) > 1
      for k in sorted(client_files.keys()):
        # Loads the file
        feature = bob.io.load( str(client_files[k]) )
        # Appends in the arrayset; assure that the data is 1-dimensional
        client_data.append(numpy.reshape(feature, feature.size))
      data.append(client_data)

    # Returns the list of Arraysets
    return data
    

  def train_projector(self, training_files, projector_file):
    """Generates the LDA projection matrix from the given features (that are sorted by identity)"""
    # Initializes an arrayset for the data
    data = self.__read_data__(training_files)

    print "Training LinearMachine using LDA"
    t = bob.trainer.FisherLDATrainer()
    self.m_machine, __eig_vals = t.train(data)
    # do not resize the machine; we will take all possible eigen vectors
    #self.m_machine.resize(self.m_machine.shape[0], self.m_config.n_outputs)
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
    return - self.m_distance_function(model, probe)

    

