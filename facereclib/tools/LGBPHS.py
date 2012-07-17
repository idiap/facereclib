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
  
  def enrol(self, enrol_features):
    """Enroling model by taking the average of all features"""
    model = numpy.zeros(enrol_features[0].shape, dtype = numpy.float64)
    # add up models
    for i in range(len(enrol_features)):
      model += enrol_features[i]
    # normalize by number of models
    model /= float(len(enrol_features))
    # return averaged model
    return model
    
  def score(self, model, probe):
    """Computes the score as the histogram intersection"""
    return self.m_factor * self.m_distance_function(model.flatten(), probe.flatten()) 


