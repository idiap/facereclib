#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy
import math

from .Tool import Tool

class GaborJets (Tool):
  """Tool chain for computing Gabor jets, Gabor graphs, and Gabor graph comparisons"""

  def __init__(
      self,
      # parameters for the tool
      gabor_jet_similarity_type,
      multiple_feature_scoring = 'max_jet',
      # some similarity functions might need a GaborWaveletTransform class, so we have to provide the parameters here as well...
      gabor_directions = 8,
      gabor_scales = 5,
      gabor_sigma = 2. * math.pi,
      gabor_maximum_frequency = math.pi / 2.,
      gabor_frequency_step = math.sqrt(.5),
      gabor_power_of_k = 0,
      gabor_dc_free = True
  ):

    # call base class constructor
    Tool.__init__(
        self,

        gabor_jet_similarity_type = str(gabor_jet_similarity_type),
        multiple_feature_scoring = multiple_feature_scoring,
        gabor_directions = gabor_directions,
        gabor_scales = gabor_scales,
        gabor_sigma = gabor_sigma,
        gabor_maximum_frequency = gabor_maximum_frequency,
        gabor_frequency_step = gabor_frequency_step,
        gabor_power_of_k = gabor_power_of_k,
        gabor_dc_free = gabor_dc_free,

        multiple_model_scoring = None,
        multiple_probe_scoring = None
    )

    # graph machine for enrolling models and comparing graphs
    self.m_graph_machine = bob.machine.GaborGraphMachine()

    # the Gabor wavelet transform; used by (some of) the Gabor jet similarities
    gwt = bob.ip.GaborWaveletTransform(
        number_of_scales = gabor_scales,
        number_of_angles = gabor_directions,
        sigma = gabor_sigma,
        k_max = gabor_maximum_frequency,
        k_fac = gabor_frequency_step,
        pow_of_k = gabor_power_of_k,
        dc_free = gabor_dc_free
    )

    # jet comparison function
    self.m_similarity_function = bob.machine.GaborJetSimilarity(gabor_jet_similarity_type, gwt)

    # how to proceed with multiple features per model
    self.m_jet_scoring = {
        'average_model' : None, # compute an average model
        'average' : numpy.average, # compute the average similarity
        'min_jet' : numpy.min, # for each jet location, compute the minimum similarity
        'max_jet' : numpy.max, # for each jet location, compute the maximum similarity
        'med_jet' : numpy.median, # for each jet location, compute the median similarity
        'min_graph' : numpy.average, # for each model graph, compute the minimum average similarity
        'max_graph' : numpy.average, # for each model graph, compute the maximum average similarity
        'med_graph' : numpy.average, # for each model graph, compute the median average similarity
    }[multiple_feature_scoring]

    self.m_graph_scoring = {
        'average_model' : None, # compute an average model
        'average' : numpy.average, # compute the average similarity
        'min_jet' : numpy.average, # for each jet location, compute the minimum similarity
        'max_jet' : numpy.average, # for each jet location, compute the maximum similarity
        'med_jet' : numpy.average, # for each jet location, compute the median similarity
        'min_graph' : numpy.min, # for each model graph, compute the minimum average similarity
        'max_graph' : numpy.max, # for each model graph, compute the maximum average similarity
        'med_graph' : numpy.median, # for each model graph, compute the median average similarity
    }[multiple_feature_scoring]


  def enroll(self, enroll_features):
    """Enrolls the model by computing an average graph for each model"""
    graph_count = len(enroll_features)

    shape = [graph_count] + list(enroll_features[0].shape)
    model = numpy.ndarray(tuple(shape), dtype=numpy.float64)
    for c, graph in enumerate(enroll_features):
      if graph.shape != model.shape[1:]:
        raise Exception('Size mismatched')
      model[c] = graph

    if self.m_jet_scoring is None:
      if model.ndim != 4:
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
    if self.m_jet_scoring is None:
      # compute sum of Gabor jet similarities between averaged model graph and probe graph
      return numpy.average([self.m_similarity_function(model[n], probe[n]) for n in range(model.shape[0])])
    else:
      # compute all Gabor jet similarities
      scores = [[self.m_similarity_function(model[c,n], probe[n]) for n in range(model.shape[1])] for c in range(model.shape[0])]
      # for each jet location, compute the desired score averaging
      return self.m_graph_scoring(self.m_jet_scoring(scores, axis=0))


  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model graph(s) and several given probe graphs."""
    if self.m_jet_scoring is None:
      # compute sum of Gabor jet similarities between averaged model graph and probe graphs
      return numpy.average([self.m_similarity_function(model[n], probes[p][n]) for n in range(model.shape[0]) for p in len(probes)])
    else:
      # compute all Gabor jet similarities
      scores = [[self.m_similarity_function(model[c,n], probes[p][n]) for n in range(model.shape[1])] for p in range(len(probes)) for c in range(model.shape[0])]
      # for each jet location, compute the desired score averaging
      return self.m_graph_scoring(self.m_jet_scoring(scores, axis=0))



