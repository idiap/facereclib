#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy

class LGBPHSTool:
  """Tool chain for computing local Gabor binary pattern histogram sequences"""

  def __init__(self, setup):
    """Initializes the local Gabor binary pattern histogram sequence tool chain with the given file selector object"""
    # nothing to be done here
    self.m_distance_function = setup.distance_function
    self.m_factor =  -1. if setup.is_distance_function else 1
    self.m_sparse = setup.SPARSE if hasattr(setup, 'SPARSE') else False

  def enroll(self, enroll_features):
    """Enrolling model by taking the average of all features"""
    if self.m_sparse:
      # get all indices for the sparse model
      values = {}
      # assert that we got sparse features
      assert enroll_features[0].shape[0] == 2
      # iterate through all sparse features
      for i in range(len(enroll_features)):
        feature = enroll_features[i]
        # collect the values by index
        for j in range(feature.shape[1]):
          index = int(feature[0,j])
          value = feature[1,j] / float(len(enroll_features))
          # add up values
          if index in values:
            values[index] += value
          else:
            values[index] = value

      # create model containing all the used indices
      model = numpy.ndarray((2, len(values)), dtype = numpy.float64)

      i = 0
      for index in sorted(values.keys()):
        model[0,i] = index
        model[1,i] = values[index]
        i += 1
    else:
      model = numpy.zeros(enroll_features[0].shape, dtype = numpy.float64)
      # add up models
      for i in range(len(enroll_features)):
        model += enroll_features[i]
      # normalize by number of models
      model /= float(len(enroll_features))

    # return averaged model
    return model


  def score(self, model, probe):
    """Computes the score using the specified histogram measure; returns a similarity value (bigger -> better)"""
    if self.m_sparse:
      return self.m_factor * self.m_distance_function(model[0,:], model[1,:], probe[0,:], probe[1,:])
    else:
      return self.m_factor * self.m_distance_function(model.flatten(), probe.flatten())


