#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.core
import bob.io.base
import bob.learn.em

import numpy
import types

from .Tool import Tool
from .UBMGMM import UBMGMM
from .. import utils


class ISV (UBMGMM):
  """Tool for computing Unified Background Models and Gaussian Mixture Models of the features"""


  def __init__(
      self,
      # ISV training
      subspace_dimension_of_u,       # U subspace dimension
      isv_training_iterations = 10,  # Number of EM iterations for the ISV training
      # ISV enrollment
      isv_enroll_iterations = 1,     # Number of iterations for the enrollment phase

      multiple_probe_scoring = None, # scoring when multiple probe files are available

      # parameters of the GMM
      **kwargs
  ):
    """Initializes the local UBM-GMM tool with the given file selector object"""
    # call base class constructor with its set of parameters
    UBMGMM.__init__(self, **kwargs)

    # call tool constructor to overwrite what was set before
    Tool.__init__(
        self,
        performs_projection = True,
        use_projected_features_for_enrollment = True,
        requires_enroller_training = False, # not needed anymore because it's done while training the projector
        split_training_features_by_client = True,

        subspace_dimension_of_u = subspace_dimension_of_u,
        isv_training_iterations = isv_training_iterations,
        isv_enroll_iterations = isv_enroll_iterations,

        multiple_model_scoring = None,
        multiple_probe_scoring = multiple_probe_scoring,
        **kwargs
    )

    self.m_subspace_dimension_of_u = subspace_dimension_of_u
    self.m_isv_training_iterations = isv_training_iterations
    self.m_isv_enroll_iterations = isv_enroll_iterations


  def _train_isv(self, data):
    """Train the ISV model given a dataset"""
    utils.info("  -> Training ISV enroller")
    self.m_isvbase = bob.learn.em.ISVBase(self.m_ubm, self.m_subspace_dimension_of_u)
    # train ISV model
    trainer = bob.learn.em.ISVTrainer(self.m_relevance_factor)
    bob.learn.em.train(trainer, self.m_isvbase, data, self.m_isv_training_iterations, rng=self.m_rng)


  def train_projector(self, train_features, projector_file):
    """Train Projector and Enroller at the same time"""

    data1 = numpy.vstack([feature for client in train_features for feature in client])

    UBMGMM._train_projector_using_array(self, data1)
    # to save some memory, we might want to delete these data
    del data1

    # project training data
    utils.info("  -> Projecting training data")
    data = []
    for client_features in train_features:
      list = []
      for feature in client_features:
        # Initializes GMMStats object
        self.m_gmm_stats = bob.learn.em.GMMStats(*self.m_ubm.shape)
        list.append(UBMGMM.project(self, feature))
      data.append(list)

    # train ISV
    self._train_isv(data)

    # Save the ISV base AND the UBM into the same file
    self.save_projector(projector_file)


  def save_projector(self, projector_file):
    """Save the GMM and the ISV model in the same HDF5 file"""
    hdf5file = bob.io.base.HDF5File(projector_file, "w")
    hdf5file.create_group('Projector')
    hdf5file.cd('Projector')
    self.m_ubm.save(hdf5file)

    hdf5file.cd('/')
    hdf5file.create_group('Enroller')
    hdf5file.cd('Enroller')
    self.m_isvbase.save(hdf5file)

  def load_isv(self, isv_file):
    hdf5file = bob.io.base.HDF5File(isv_file)
    self.m_isvbase = bob.learn.em.ISVBase(hdf5file)
    # add UBM model from base class
    self.m_isvbase.ubm = self.m_ubm

  def load_projector(self, projector_file):
    """Load the GMM and the ISV model from the same HDF5 file"""
    hdf5file = bob.io.base.HDF5File(projector_file)

    # Load Projector
    hdf5file.cd('/Projector')
    self.load_ubm(hdf5file)

    # Load Enroller
    hdf5file.cd('/Enroller')
    self.load_isv(hdf5file)

    self.m_machine = bob.learn.em.ISVMachine(self.m_isvbase)
    self.m_trainer = bob.learn.em.ISVTrainer(self.m_relevance_factor)
    self.m_rng = bob.core.random.mt19937(self.m_init_seed)


  #######################################################
  ################ ISV training #########################
  def project_isv(self, projected_ubm):
    projected_isv = numpy.ndarray(shape=(self.m_ubm.shape[0]*self.m_ubm.shape[1],), dtype=numpy.float64)
    model = bob.learn.em.ISVMachine(self.m_isvbase)
    model.estimate_ux(projected_ubm, projected_isv)
    return projected_isv

  def project(self, feature_array):
    """Computes GMM statistics against a UBM, then corresponding Ux vector"""
    projected_ubm = UBMGMM.project(self,feature_array)
    projected_isv = self.project_isv(projected_ubm)
    return [bob.learn.em.GMMStats(projected_ubm), projected_isv]

  #######################################################
  ################## ISV model enroll ####################

  def save_feature(self, data, feature_file):
    gmmstats = data[0]
    Ux = data[1]
    hdf5file = bob.io.base.HDF5File(feature_file, "w") if isinstance(feature_file, str) else feature_file
    hdf5file.create_group('gmmstats')
    hdf5file.cd('gmmstats')
    gmmstats.save(hdf5file)
    hdf5file.cd('..')
    hdf5file.set('Ux', Ux)

  def read_feature(self, feature_file):
    """Read the type of features that we require, namely GMMStats"""
    hdf5file = bob.io.base.HDF5File(feature_file)
    hdf5file.cd('gmmstats')
    gmmstats = bob.learn.em.GMMStats(hdf5file)
    return gmmstats


  def enroll(self, enroll_features):
    """Performs ISV enrollment"""
    self.m_trainer.enroll(self.m_machine, enroll_features, self.m_isv_enroll_iterations)
    # return the resulting gmm
    return self.m_machine


  ######################################################
  ################ Feature comparison ##################
  def read_model(self, model_file):
    """Reads the ISV Machine that holds the model"""
    machine = bob.learn.em.ISVMachine(bob.io.base.HDF5File(model_file))
    machine.isv_base = self.m_isvbase
    return machine

  def read_probe(self, probe_file):
    """Read the type of features that we require, namely GMMStats"""
    hdf5file = bob.io.base.HDF5File(probe_file)
    hdf5file.cd('gmmstats')
    gmmstats = bob.learn.em.GMMStats(hdf5file)
    hdf5file.cd('..')
    Ux = hdf5file.read('Ux')
    return [gmmstats, Ux]

  def score(self, model, probe):
    """Computes the score for the given model and the given probe."""
    gmmstats = probe[0]
    Ux = probe[1]
    return model.forward_ux(gmmstats, Ux)

  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and several given probe files."""
    if self.m_probe_fusion_function is not None:
      # When a multiple probe fusion function is selected, use it
      return Tool.score_for_multiple_probes(self, model, probes)
    else:
      # Otherwise: compute joint likelihood of all probe features
      # create GMM statistics from first probe statistics
      gmmstats_acc = bob.learn.em.GMMStats(probes[0][0])
      # add all other probe statistics
      for i in range(1,len(probes)):
        gmmstats_acc += probes[i][0]
      # compute ISV score with the accumulated statistics
      projected_isv_acc = numpy.ndarray(shape=(self.m_ubm.shape[0]*self.m_ubm.shape[1],), dtype=numpy.float64)
      model.estimate_ux(gmmstats_acc, projected_isv_acc)
      return model.forward_ux(gmmstats_acc, projected_isv_acc)
