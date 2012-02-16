#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy


class PCATool:
  """Tool chain for computing Unified Background Models and Gaussian Mixture Models of the features"""

  def __init__(self, file_selector, setup):
    """Initializes the local UBM-GMM tool chain with the given file selector object"""
    self.m_config = setup
    self.m_machine = None
    self.m_distance_function = self.m_config.distance_function
    

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
    T = bob.trainer.SVDPCATrainer()
    self.m_machine, __eig_vals = T.train(data)
    # Machine: get shape, then resize
    self.m_machine.resize(self.m_machine.shape[0], self.m_config.n_outputs)
    self.m_machine.save(bob.io.HDF5File(projector_file))


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

    

