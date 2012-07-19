#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy
import math

class BICTool:
  """Computes the Intrapersonal/Extrapersonal classifier"""

  def sqr(self, x):
    return x*x
  
  def __init__(self, setup):
    self.m_cfg = setup
    self.m_distance_function = setup.distance_function
    self.m_maximum_pair_count = setup.maximum_pair_count if hasattr(setup, 'maximum_pair_count') else None
    self.m_use_dffs = setup.USE_DFFS if hasattr(setup, 'USE_DFFS') else False
    if hasattr(setup, 'M_I') and hasattr(setup, 'M_E'):
      self.m_M_I = setup.M_E
      self.m_M_E = setup.M_E
      self.m_bic_machine = bob.machine.BICMachine(self.m_use_dffs)
    else:
      self.m_bic_machine = bob.machine.BICMachine(False)
    
  def __compare__(self, feature_1, feature_2):
    """Computes a vector of similarities"""
    assert feature_1.shape == feature_2.shape
    sim = numpy.ndarray((feature_1.shape[0],), dtype = numpy.float64)
    for i in range(feature_1.shape[0]):
      sim[i] = self.m_distance_function(feature_1[i], feature_2[i])
    return sim

  def __intra_extra_pairs__(self, train_files):
    """Computes intrapersonal and extrapersonal pairs of features from given training files"""
    # first, load all features
    features = {}
    for id in sorted(train_files.keys()):
      features[id] = []
      for im in train_files[id]:
        feature = bob.io.load(str(train_files[id][im]))
        features[id].append(feature)
  
    # generate intrapersonal pairs
    intra_pairs = []
    for id in sorted(features.keys()):
      for i in range(len(features[id])-1):
        for j in range (i+1, len(features[id])):
          intra_pairs.append((features[id][i], features[id][j]))
          
    # generate extrapersonal pairs
    extra_pairs = []
    for id in sorted(features.keys()):
      for i in range(len(features[id])):
        for jd in sorted(features.keys()):
          if id != jd:
            for j in range(len(features[jd])):
              extra_pairs.append((features[id][i], features[jd][j]))
              
    # limit the number of pairs by random selection 
    if self.m_maximum_pair_count != None:
      if len(intra_pairs) > self.m_maximum_pair_count:
        print "Limiting intrapersonal pairs from", len(intra_pairs), "to", self.m_maximum_pair_count
        numpy.random.shuffle(intra_pairs)
        intra_pairs = intra_pairs[:self.m_maximum_pair_count]
      if len(extra_pairs) > self.m_maximum_pair_count:
        print "Limiting extrapersonal pairs from", len(extra_pairs), "to", self.m_maximum_pair_count
        numpy.random.shuffle(extra_pairs)
        extra_pairs = extra_pairs[:self.m_maximum_pair_count]
    
    return (intra_pairs, extra_pairs)
    
  def __trainset_for__(self, pairs):
    """Computes the bob.io.Arrayset containing the comparison results for the given set of image pairs."""
    comparison_results = bob.io.Arrayset()
    for (f1, f2) in pairs:
      comparison_results.append(self.__compare__(f1, f2))
    return comparison_results
  
  def train_enroler(self, train_files, enroler_file):
    """Trains the IEC Tool, i.e., computes intrapersonal and extrapersonal subspaces"""

    # compute intrapersonal and extrapersonal pairs
    intra_pairs, extra_pairs = self.__intra_extra_pairs__(train_files)
    
    # train the BIC Machine with these pairs
    print "Computing", len(intra_pairs), "intrapersonal results"
    intra_vectors = self.__trainset_for__(intra_pairs)
    print "Computing", len(extra_pairs), "extrapersonal results"
    extra_vectors = self.__trainset_for__(extra_pairs)
    
    print "Training BIC machine"
    trainer = bob.trainer.BICTrainer(self.m_M_I, self.m_M_E) if self.m_M_I != None else bob.trainer.BICTrainer()
    trainer.train(self.m_bic_machine, intra_vectors, extra_vectors)
    
    # save the machine to file
    self.m_bic_machine.save(bob.io.HDF5File(enroler_file, 'w'))
    
  def load_enroler(self, enroler_file):
    """Reads the intrapersonal and extrapersonal mean and variance values""" 
    self.m_bic_machine.load(bob.io.HDF5File(enroler_file, 'r'))
    # to set this should not be required, but just in case 
    # you re-use a trained enroler file that hat different setup of use_DFFS    
    self.m_bic_machine.use_dffs = self.m_use_dffs
    
  def enrol(self, enrol_features):
    """Enrols features by concatenating them"""
    shape = list(enrol_features[0].shape)
    shape.insert(0,len(enrol_features))
    model = numpy.ndarray(tuple(shape), dtype = numpy.float64)
    
    for i in range(len(enrol_features)):
      model[i] = enrol_features[i]
    
    return model
  
  def __iec_score__(self, feature_1, feature_2):
    """Computes the IEC score for two features"""
    # compute similarity vector
    distance_vector = self.__compare__(feature_1, feature_2)
    
    # apply the BIC machine
    return self.m_bic_machine(distance_vector)
    
  
  def score(self, model, probe):
    """Computes the IEC score for the given model and probe pair"""
    # compute average score for the models
    s = 0
    for i in range(model.shape[0]):
      s += self.__iec_score__(model[0], probe)
      
    return s / model.shape[0] 
