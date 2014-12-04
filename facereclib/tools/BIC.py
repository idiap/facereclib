#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.io.base
import bob.learn.linear

import numpy
import math

from .Tool import Tool
from .. import utils

class BIC (Tool):
  """Computes the Intrapersonal/Extrapersonal classifier"""

  def sqr(self, x):
    return x*x

  def __init__(
      self,
      comparison_function, # the function to be used to compare two features; this highly depends on the type of features that are used
      maximum_training_pair_count = None,  # if set, limit the number of training pairs to the given number in a non-random manner
      subspace_dimensions = None, # if set as a pair (intra_dim, extra_dim), PCA subspace truncation for the two classes is performed
      uses_dffs = False, # use the distance from feature space; only valid when PCA truncation is enabled; WARNING: uses this flag with care
      load_function = utils.load,
      save_function = utils.save,
      **kwargs # parameters directly sent to the base class
  ):

    # call base class function and register that this tool requires training for the enrollment
    Tool.__init__(
        self,
        requires_enroller_training = True,

        comparison_function = str(comparison_function),
        maximum_training_pair_count = maximum_training_pair_count,
        subspace_dimensions = subspace_dimensions,
        uses_dffs = uses_dffs,
        load_function=str(load_function),
        save_function=str(save_function),

        **kwargs
    )

    # set up the BIC tool
    self.m_comparison_function = comparison_function
    self.m_load_function = load_function
    self.m_save_function = save_function
    self.m_maximum_pair_count = maximum_training_pair_count
    self.m_use_dffs = uses_dffs
    if subspace_dimensions is not None:
      self.m_M_I = subspace_dimensions[0]
      self.m_M_E = subspace_dimensions[1]
      self.m_bic_machine = bob.learn.linear.BICMachine(self.m_use_dffs)
    else:
      self.m_bic_machine = bob.learn.linear.BICMachine(False)
      self.m_M_I = None
      self.m_M_E = None


  def __compare__(self, feature_1, feature_2):
    """Computes a vector of similarities"""
    return self.m_comparison_function(feature_1, feature_2)


  def __intra_extra_pairs__(self, train_features):
    """Computes intrapersonal and extrapersonal pairs of features from given training files"""
    # generate intrapersonal pairs
    intra_pairs = []
    for client in range(len(train_features)):
      for c in range(len(train_features[client])-1):
        for c2 in range (c+1, len(train_features[client])):
          intra_pairs.append((train_features[client][c], train_features[client][c2]))

    # generate extrapersonal pairs
    extra_pairs = []
    for client in range(len(train_features)):
      for c in range(len(train_features[client])):
        for impostor in range(len(train_features)):
          if client != impostor:
            for i in range(len(train_features[impostor])):
              extra_pairs.append((train_features[client][c], train_features[impostor][i]))

    # limit the number of pairs by random selection
    if self.m_maximum_pair_count != None:
      if len(intra_pairs) > self.m_maximum_pair_count:
        utils.info("  -> Limiting intrapersonal pairs from %d to %d" %(len(intra_pairs),self.m_maximum_pair_count))
        intra_pairs = [intra_pairs[i] for i in utils.quasi_random_indices(len(intra_pairs), self.m_maximum_pair_count)]
      if len(extra_pairs) > self.m_maximum_pair_count:
        utils.info("  -> Limiting extrapersonal pairs from %d to %d" %(len(extra_pairs), self.m_maximum_pair_count))
        extra_pairs = [extra_pairs[i] for i in utils.quasi_random_indices(len(extra_pairs), self.m_maximum_pair_count)]

    return (intra_pairs, extra_pairs)


  def __trainset_for__(self, pairs):
    """Computes the array containing the comparison results for the given set of image pairs."""
    return numpy.vstack([self.__compare__(f1, f2) for (f1, f2) in pairs])


  def train_enroller(self, train_features, enroller_file):
    """Trains the IEC Tool, i.e., computes intrapersonal and extrapersonal subspaces"""

    # compute intrapersonal and extrapersonal pairs
    utils.info("  -> Computing pairs")
    intra_pairs, extra_pairs = self.__intra_extra_pairs__(train_features)

    # train the BIC Machine with these pairs
    utils.info("  -> Computing %d intrapersonal results" % len(intra_pairs))
    intra_vectors = self.__trainset_for__(intra_pairs)
    utils.info("  -> Computing %d extrapersonal results" % len(extra_pairs))
    extra_vectors = self.__trainset_for__(extra_pairs)

    utils.info("  -> Training BIC machine")
    trainer = bob.learn.linear.BICTrainer(self.m_M_I, self.m_M_E) if self.m_M_I != None else bob.learn.linear.BICTrainer()
    trainer.train(intra_vectors, extra_vectors, self.m_bic_machine)

    # save the machine to file
    self.m_bic_machine.save(bob.io.base.HDF5File(enroller_file, 'w'))


  def load_enroller(self, enroller_file):
    """Reads the intrapersonal and extrapersonal mean and variance values"""
    self.m_bic_machine.load(bob.io.base.HDF5File(enroller_file, 'r'))
    # to set this should not be required, but just in case
    # you re-use a trained enroller file that hat different setup of use_DFFS
    self.m_bic_machine.use_DFFS = self.m_use_dffs


  def enroll(self, enroll_features):
    """Enrolls features by concatenating them"""
    return enroll_features


  def save_model(self, model, model_file):
    """Writes all features of the model into one HDF5 file, using the ``save_function`` specified in the constructor."""
    hdf5 = bob.io.base.HDF5File(model_file, "w")
    for i, f in enumerate(model):
      hdf5.create_group("Feature%d" % i)
      hdf5.cd("Feature%d" % i)
      self.m_save_function(f, hdf5)
      hdf5.cd("..")


  def read_model(self, model_file):
    """Loads all features of the model from the HDF5 file, using the ``load_function`` specified in the constructor."""
    hdf5 = bob.io.base.HDF5File(model_file)
    i = 0
    model = []
    while hdf5.has_group("Feature%d" % i):
      hdf5.cd("Feature%d" % i)
      model.append(self.m_load_function(hdf5))
      hdf5.cd("..")
      i += 1
    return model


  def read_probe(self, probe_file):
    """Loads the probe feature from file, using the ``load_function`` specified in the constructor."""
    return self.m_load_function(bob.io.base.HDF5File(probe_file))


  def __iec_score__(self, feature_1, feature_2):
    """Computes the IEC score for two features"""
    # compute similarity vector
    distance_vector = self.__compare__(feature_1, feature_2)

    # apply the BIC machine
    return self.m_bic_machine(distance_vector)

  def score(self, model, probe):
    """Computes the IEC score for the given model and probe pair"""
    # compute average score for the models
    return self.m_model_fusion_function([self.__iec_score__(model[i], probe) for i in range(len(model))])
