#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy

class GaborJetTool:
  """Tool chain for computing Gabor jets, Gabor graphs, and Gabor graph comparisons"""
  
  def __init__(self, setup):
    # graph machine for enroling models and comparing graphs
    self.m_graph_machine = bob.machine.GaborGraphMachine()
    # jet comparison function
    self.m_similarity_function = setup.jet_similarity_function
    self.m_average_model = setup.extract_averaged_model

    
  def enrol(self, enrol_features):
    """Enrols the model by computing an average graph for each model"""
    graph_count = len(enrol_features)
    
    c = 0 # counts the number of enrolment files
    shape = list(enrol_features[0].shape)
    shape.insert(0, graph_count)
    model = numpy.ndarray(tuple(shape), dtype=numpy.float64)
    for graph in enrol_features:
      if graph.shape != model.shape[1:]:
        raise Exception('Size mismatched')

      model[c] = graph
      c += 1

    if self.m_average_model:
      if model.ndim() != 4:
        raise Exception('Averaging is only supported when phases are included')
      # compute average model
      average = numpy.ndarray(model.shape[1:], dtype=numpy.float64)
      self.m_graph_machine.average(model, average)
      # return the average
      return average
      
    else: 
      # return the generated model
      return model

  
  def score(self, model, probe):
    """Computes the score of the probe and the model"""
    return self.m_graph_machine.similarity(model, probe, self.m_similarity_function)
   


