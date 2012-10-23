#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy

from .Tool import Tool
from . import UBMGMMTool


class JFATool (UBMGMMTool):
  """Tool chain for computing Unified Background Models and Gaussian Mixture Models of the features"""

  def __init__(
      self,
      # JFA training
      subspace_dimension_of_u,       # U subspace dimension
      subspace_dimension_of_v,       # V subspace dimension
      jfa_training_iterations = 10,  # Number of EM iterations for the JFA training
      # JFA enrollment
      jfa_enroll_iterations = 1,     # Number of iterations for the enrollment phase
      # parameters of the GMM
      **kwargs
  ):
    """Initializes the local UBM-GMM tool with the given file selector object"""
    # call base class constructor
    UBMGMMTool.__init__(self, **kwargs)

    # call tool constructor to overwrite what was set before
    Tool.__init__(
        self,
        performs_projection = True,
        use_projected_features_for_enrollment = True,
        requires_enroller_training = True
    )

    self.m_subspace_dimension_of_u = subspace_dimension_of_u
    self.m_subspace_dimension_of_v = subspace_dimension_of_v
    self.m_jfa_training_iterations = jfa_training_iterations
    self.m_jfa_enroll_iterations = jfa_enroll_iterations

  # Here, we just need to load the UBM from the projector file.
  def load_projector(self, projector_file):
    """Reads the UBM model from file"""
    # read UBM
    self.m_ubm = bob.machine.GMMMachine(bob.io.HDF5File(projector_file))
    self.m_ubm.set_variance_thresholds(self.m_variance_threshold)
    # Initializes GMMStats object
    self.m_gmm_stats = bob.machine.GMMStats(self.m_ubm.dim_c, self.m_ubm.dim_d)



  #######################################################
  ################ JFA training #########################
  def train_enroller(self, train_features, enroller_file):
    # create a JFABasemachine with the UBM from the base class
    self.m_jfabase = bob.machine.JFABaseMachine(self.m_ubm, self.m_subspace_dimension_of_u, self.m_subspace_dimension_of_v)
    self.m_jfabase.ubm = self.m_ubm

    # train the JFA
    t = bob.trainer.JFABaseTrainer(self.m_jfabase)
    t.train(train_features, self.m_jfa_training_iterations)

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
    """Enrolls a GMM using MAP adaptation"""

    self.m_trainer.enrol(enroll_features, self.m_jfa_enroll_iterations)
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
    """Computes the score for the given model and the given probe"""
    scores = numpy.ndarray((1,), 'float64')
    model.forward([probe], scores)
    return scores[0]

