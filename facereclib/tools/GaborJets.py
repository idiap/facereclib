#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.ip.gabor
import bob.io.base

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

        gabor_jet_similarity_type = gabor_jet_similarity_type,
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

    # the Gabor wavelet transform; used by (some of) the Gabor jet similarities
    gwt = bob.ip.gabor.Transform(
        number_of_scales = gabor_scales,
        number_of_directions = gabor_directions,
        sigma = gabor_sigma,
        k_max = gabor_maximum_frequency,
        k_fac = gabor_frequency_step,
        power_of_k = gabor_power_of_k,
        dc_free = gabor_dc_free
    )

    # jet comparison function
    self.m_similarity_function = bob.ip.gabor.Similarity(gabor_jet_similarity_type, gwt)

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
    assert len(enroll_features)
    if self.m_jet_scoring is not None:
      return enroll_features

    # compute average model
    return [bob.ip.gabor.Jet([enroll_features[g][n] for g in range(len(enroll_features))], normalize=True) for n in range(len(enroll_features[0]))]


  def save_model(self, model, model_file):
    f = bob.io.base.HDF5File(model_file, 'w')
    if self.m_jet_scoring is None:
      # only one averaged model
      bob.ip.gabor.save_jets(model, f)
    else:
      # several model graphs
      f.set("NumberOfModels", len(model))
      for g in range(len(model)):
        name = "Model" + str(g+1)
        f.create_group(name)
        f.cd(name)
        bob.ip.gabor.save_jets(model[g], f)
        f.cd("..")
    f.close()

  def read_model(self, model_file):
    f = bob.io.base.HDF5File(model_file)
    if self.m_jet_scoring is None:
      # only one graph
      assert not f.has_key("NumberOfModels")
      return bob.ip.gabor.load_jets(f)
    else:
      # several graphs
      count = f.get("NumberOfModels")
      model = []
      for g in range(count):
        name = "Model" + str(g+1)
        f.cd(name)
        model.append(bob.ip.gabor.load_jets(f))
        f.cd("..")
      return model

  def read_probe(self, probe_file):
    return bob.ip.gabor.load_jets(bob.io.base.HDF5File(probe_file))

  def score(self, model, probe):
    """Computes the score of the probe and the model"""
    if self.m_jet_scoring is None:
      # compute sum of Gabor jet similarities between averaged model graph and probe graph
      return numpy.average([self.m_similarity_function(model[n], probe[n]) for n in range(len(model))])
    else:
      # compute all Gabor jet similarities
      scores = [[self.m_similarity_function(model[c][n], probe[n]) for n in range(len(model[0]))] for c in range(len(model))]
      # for each jet location, compute the desired score averaging
      return self.m_graph_scoring(self.m_jet_scoring(scores, axis=0))


  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model graph(s) and several given probe graphs."""
    if self.m_jet_scoring is None:
      # compute sum of Gabor jet similarities between averaged model graph and probe graphs
      return numpy.average([self.m_similarity_function(model[n], probes[p][n]) for n in range(len(model)) for p in range(len(probes))])
    else:
      # compute all Gabor jet similarities
      scores = [[self.m_similarity_function(model[c][n], probes[p][n]) for n in range(len(model[0]))] for p in range(len(probes)) for c in range(len(model))]
      # for each jet location, compute the desired score averaging
      return self.m_graph_scoring(self.m_jet_scoring(scores, axis=0))



