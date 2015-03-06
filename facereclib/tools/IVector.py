#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import bob.core
import bob.io.base
import bob.learn.linear
import bob.learn.em

import numpy

from .Tool import Tool
from .UBMGMM import UBMGMM
from .. import utils

class IVector (UBMGMM):
  """Tool for extracting I-Vectors"""

  def __init__(
      self,
      # IVector training
      subspace_dimension_of_t,       # T subspace dimension
      update_sigma = True,
      tv_training_iterations = 25,  # Number of EM iterations for the JFA training
      variance_threshold = 1e-5,
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
        split_training_features_by_client = False,

        subspace_dimension_of_t = subspace_dimension_of_t,
        update_sigma = update_sigma,
        tv_training_iterations = tv_training_iterations,
        variance_threshold = variance_threshold,

        multiple_model_scoring = None,
        multiple_probe_scoring = None,
        **kwargs
    )

    self.m_update_sigma = update_sigma
    self.m_subspace_dimension_of_t = subspace_dimension_of_t
    self.m_tv_training_iterations = tv_training_iterations
    self.m_variance_threshold = variance_threshold


  def _train_ivector(self, train_features):
    utils.info("  -> Projecting training data")
    data = []
    for feature in train_features:
      # Initializes GMMStats object
      self.m_gmm_stats = bob.learn.em.GMMStats(*self.m_ubm.shape)
      data.append(UBMGMM.project(self, feature))

    utils.info("  -> Training IVector enroller")
    self.m_tv = bob.learn.em.IVectorMachine(self.m_ubm, self.m_subspace_dimension_of_t)
    self.m_tv.variance_threshold = self.m_variance_threshold

    # train IVector model
    trainer = bob.learn.em.IVectorTrainer(update_sigma=self.m_update_sigma)
    bob.learn.em.train(trainer, self.m_tv, data, self.m_tv_training_iterations, rng=bob.core.random.mt19937(self.m_init_seed))

    return data

  def _train_whitening(self, training_features):
   # load GMM stats from training files
    ivectors_matrix = numpy.vstack(training_features)
    # create a Linear Machine
    self.m_whitening_machine = bob.learn.linear.Machine(ivectors_matrix.shape[1],ivectors_matrix.shape[1])
    # create the whitening trainer
    t = bob.learn.linear.WhiteningTrainer()

    t.train(ivectors_matrix, self.m_whitening_machine)

  def train_projector(self, train_features, projector_file):
    """Train Projector and Enroller at the same time"""

    data = numpy.vstack(train_features)

    UBMGMM._train_projector_using_array(self, data)
    # to save some memory, we might want to delete these data
    del data

    # train IVector
    training_gmms = self._train_ivector(train_features)

    # project training i-vectors
    whitening_train_data = [self.project_ivec(gmm) for gmm in training_gmms]
    self._train_whitening(whitening_train_data)

    # save
    self.save_projector(projector_file)

  def save_projector(self, projector_file):
    # Save the IVector base AND the UBM AND the whitening into the same file
    hdf5file = bob.io.base.HDF5File(projector_file, "w")
    hdf5file.create_group('Projector')
    hdf5file.cd('Projector')
    self.m_ubm.save(hdf5file)

    hdf5file.cd('/')
    hdf5file.create_group('Enroller')
    hdf5file.cd('Enroller')
    self.m_tv.save(hdf5file)

    hdf5file.cd('/')
    hdf5file.create_group('Whitening')
    hdf5file.cd('Whitening')
    self.m_whitening_machine.save(hdf5file)


  def load_ubm(self, ubm_file):
    hdf5file = bob.io.base.HDF5File(ubm_file)
    # read UBM
    self.m_ubm = bob.learn.em.GMMMachine(hdf5file)
    self.m_ubm.set_variance_thresholds(self.m_variance_threshold)
    # Initializes GMMStats object
    self.m_gmm_stats = bob.learn.em.GMMStats(*self.m_ubm.shape)

  def load_tv(self, tv_file):
    hdf5file = bob.io.base.HDF5File(tv_file)
    self.m_tv = bob.learn.em.IVectorMachine(hdf5file)
    # add UBM model from base class
    self.m_tv.ubm = self.m_ubm

  def load_whitening(self, whitening_file):
    hdf5file = bob.io.base.HDF5File(whitening_file)
    self.m_whitening_machine = bob.learn.linear.Machine(hdf5file)

  def load_projector(self, projector_file):
    """Load the GMM and the ISV model from the same HDF5 file"""
    hdf5file = bob.io.base.HDF5File(projector_file)

    # Load Projector
    hdf5file.cd('/Projector')
    self.load_ubm(hdf5file)

    # Load Enroller
    hdf5file.cd('/Enroller')
    self.load_tv(hdf5file)

    # Load Whitening
    hdf5file.cd('/Whitening')
    self.load_whitening(hdf5file)

  def project_ubm(self, features):
    return UBMGMM.project(self,features)

  def project_ivec(self, gmm_stats):
    return self.m_tv.project(gmm_stats)

  def project_whitening(self, ivector):
    whitened = self.m_whitening_machine.forward(ivector)
    return whitened / numpy.linalg.norm(whitened)

  #######################################################
  ############## IVector projection #####################
  def project(self, feature_array):
    """Computes GMM statistics against a UBM, then corresponding Ux vector"""
    # project UBM
    projected_ubm = self.project_ubm(feature_array)
    # project I-Vector
    ivector = self.project_ivec(projected_ubm)
    # whiten I-Vector
    return self.project_whitening(ivector)

  #######################################################
  ################## ISV model enroll ####################
  def save_feature(self, data, feature_file):
    """Saves the feature, which is the (whitened) I-Vector."""
    utils.save(data, feature_file)

  def read_feature(self, feature_file):
    """Read the type of features that we require, namely i-vectors (stored as simple numpy arrays)"""
    return utils.load(feature_file)



  #######################################################
  ################## Model  Enrollment ###################
  def enroll(self, enroll_features):
    """Performs IVector enrollment"""
    model = numpy.mean(numpy.vstack(enroll_features), axis=0)
    return model


  ######################################################
  ################ Feature comparison ##################
  def read_model(self, model_file):
    """Reads the whitened i-vector that holds the model"""
    return utils.load(model_file)

  def read_probe(self, probe_file):
    """read probe file which is an i-vector"""
    return utils.load(probe_file)

  def score(self, model, probe):
    """Computes the score for the given model and the given probe."""
    a = model/numpy.linalg.norm(model)
    b = probe/numpy.linalg.norm(probe)
    if len(a) != len(b):
        raise ValueError("a and b must be same length")
    numerator = sum(tup[0] * tup[1] for tup in zip(a,b))
    return numerator


  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and several given probe files."""
    probes = numpy.vstack([numpy.mean(numpy.vstack(probes), axis=0)])
    return self.score(model,probes)
