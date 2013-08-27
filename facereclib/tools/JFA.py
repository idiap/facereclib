#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy

from .Tool import Tool
from . import UBMGMM
from .. import utils


class JFA (UBMGMM):
  """Tool for computing Unified Background Models and Gaussian Mixture Models of the features and project it via JFA"""

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
    UBMGMM.__init__(self, **kwargs)

    # call tool constructor to overwrite what was set before
    Tool.__init__(
        self,
        performs_projection = True,
        use_projected_features_for_enrollment = True,
        requires_enroller_training = True,

        subspace_dimension_of_u = subspace_dimension_of_u,
        subspace_dimension_of_v = subspace_dimension_of_v,
        jfa_training_iterations = jfa_training_iterations,
        jfa_enroll_iterations = jfa_enroll_iterations,

        multiple_model_scoring = None,
        multiple_probe_scoring = None,
        **kwargs
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
    self.m_jfabase = bob.machine.JFABase(self.m_ubm, self.m_subspace_dimension_of_u, self.m_subspace_dimension_of_v)

    # train the JFA
    t = bob.trainer.JFATrainer(self.m_jfa_training_iterations)
    t.rng = bob.core.random.mt19937(self.m_init_seed)
    t.train(self.m_jfabase, train_features)

    # Save the JFA base AND the UBM into the same file
    self.m_jfabase.save(bob.io.HDF5File(enroller_file, "w"))



  #######################################################
  ################## JFA model enroll ####################
  def load_enroller(self, enroller_file):
    """Reads the UBM model from file"""
    # now, load the JFA base, if it is included in the file
    self.m_jfabase = bob.machine.JFABase(bob.io.HDF5File(enroller_file))
    # add UBM model from base class
    self.m_jfabase.ubm = self.m_ubm

    self.m_machine = bob.machine.JFAMachine(self.m_jfabase)
    self.m_trainer = bob.trainer.JFATrainer()
    self.m_trainer.rng = bob.core.random.mt19937(self.m_init_seed)


  def read_feature(self, feature_file):
    """Reads the projected feature to be enrolled as a model"""
    return bob.machine.GMMStats(bob.io.HDF5File(str(feature_file)))


  def enroll(self, enroll_features):
    """Enrolls a GMM using MAP adaptation"""

    self.m_trainer.enrol(self.m_machine, enroll_features, self.m_jfa_enroll_iterations)
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
    return model.forward(probe)

  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and several given probe files."""
    # TODO: Check if this is correct
    utils.warn("This function needs to be verified!")
    raise NotImplementedError('Multiple probes is not yet supported')
    #scores = numpy.ndarray((len(probes),), 'float64')
    #model.forward(probes, scores)
    #return scores[0]


