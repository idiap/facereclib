#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy
import types

from .Tool import Tool
from . import UBMGMMTool


class ISVTool (UBMGMMTool):
  """Tool chain for computing Unified Background Models and Gaussian Mixture Models of the features"""


  def __init__(self, setup):
    """Initializes the local UBM-GMM tool with the given file selector object"""
    # call base class constructor
    UBMGMMTool.__init__(self, setup)

    # call tool constructor to overwrite what was set before
    Tool.__init__(self,
                  performs_projection = True,
                  use_projected_features_for_enrollment = True,
                  requires_enroller_training = True
                  )


  def __load_gmm_stats__(self, l_files):
    """Loads a dictionary of GMM statistics from a list of filenames"""
    gmm_stats = []
    for k in l_files:
      # Processes one file
      stats = l_files[k]
      # Appends in the list
      gmm_stats.append(stats)
    return gmm_stats


  def __load_gmm_stats_list__(self, ld_files):
    """Loads a list of lists of GMM statistics from a list of dictionaries of filenames
       There is one list for each identity"""
    # Initializes a python list for the GMMStats
    gmm_stats = []
    for k in sorted(ld_files.keys()):
      # Loads the list of GMMStats for the given client
      gmm_stats_c = self.__load_gmm_stats__(ld_files[k])
      # Appends to the main list
      gmm_stats.append(gmm_stats_c)
    return gmm_stats


  # Here, we just need to load the UBM from the projector file.
  def load_projector(self, projector_file):
    """Reads the UBM model from file"""
    # read UBM
    self.m_ubm = bob.machine.GMMMachine(bob.io.HDF5File(projector_file))
    self.m_ubm.set_variance_thresholds(self.m_config.JFA_VARIANCE_THRESHOLD)
    # Initializes GMMStats object
    self.m_gmm_stats = bob.machine.GMMStats(self.m_ubm.dim_c, self.m_ubm.dim_d)


  #######################################################
  ################ ISV training #########################
  def train_enroller(self, train_features, enroller_file):
    # create a JFABasemachine with the UBM from the base class
    self.m_jfabase = bob.machine.JFABaseMachine(self.m_ubm, self.m_config.SUBSPACE_DIMENSION_OF_U)
    self.m_jfabase.ubm = self.m_ubm

    # load GMM stats from training files
    gmm_stats = self.__load_gmm_stats_list__(train_features)

    t = bob.trainer.JFABaseTrainer(self.m_jfabase)
    t.train_isv(gmm_stats, self.m_config.JFA_TRAINING_ITERATIONS, self.m_config.RELEVANCE_FACTOR)

    # Save the JFA base AND the UBM into the same file
    self.m_jfabase.save(bob.io.HDF5File(enroller_file, "w"))



  #######################################################
  ################## JFA model enroll ####################
  def load_enroller(self, enroller_file):
    """Reads the UBM model from file"""
    # now, load the JFA base, if it is included in the file
    self.m_jfabase = bob.machine.JFABaseMachine(bob.io.HDF5File(enroller_file))
    # add UBM model from base class
    self.m_jfabase.ubm = self.m_ubm

    self.m_machine = bob.machine.JFAMachine(self.m_jfabase)
    self.m_base_trainer = bob.trainer.JFABaseTrainer(self.m_jfabase)
    self.m_trainer = bob.trainer.JFATrainer(self.m_machine, self.m_base_trainer)


  def read_feature(self, feature_file):
    """Reads the projected feature to be enrolled as a model"""
    return bob.machine.GMMStats(bob.io.HDF5File(str(feature_file)))


  def enroll(self, enroll_features):
    """Performs ISV enrollment"""
    self.m_trainer.enrol(enroll_features, self.m_config.JFA_ENROLL_ITERATIONS)
    # return the resulting gmm
    return self.m_machine


  ######################################################
  ################ Feature comparison ##################
  def read_model(self, model_file):
    """Reads the JFA Machine that holds the model"""
    machine = bob.machine.JFAMachine(bob.io.HDF5File(model_file))
    machine.jfa_base = self.m_jfabase
    return machine

  read_probe = read_feature

  def score(self, model, probe):
    """Computes the score for the given model and the given probe using the scoring function from the config file"""
    scores = numpy.ndarray((1,), 'float64')
    model.forward([probe], scores)
    return scores[0]

