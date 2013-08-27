#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy
import types

from .Tool import Tool
from .UBMGMM import UBMGMM, UBMGMMVideo
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
      # Parameters when splitting GMM and ISV files
      gmm_isv_split = False,
      projected_toreplace = 'projected', # 'Magic' string in path that will be replaced by the GMM or ISV one
      projected_gmm = 'projected_gmm', # subdirectory for the projected gmm
      projected_isv = 'projected_isv', # subdirectory for the projected isv
      projector_toreplace = 'Projector.hdf5', # 'Magic' string in path that will be replaced by the GMM or ISV one
      gmm_filename = 'gmm.hdf5', # filename for the GMM model
      isv_filename = 'isv.hdf5', # filename for the ISV model
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
        gmm_isv_split = gmm_isv_split,
        projected_toreplace = projected_toreplace,
        projected_gmm =projected_gmm,
        projected_isv = projected_isv,
        projector_toreplace = projector_toreplace,
        gmm_filename = gmm_filename,
        isv_filename = isv_filename,

        multiple_model_scoring = None,
        multiple_probe_scoring = None,
        **kwargs
    )

    self.m_subspace_dimension_of_u = subspace_dimension_of_u
    self.m_isv_training_iterations = isv_training_iterations
    self.m_isv_enroll_iterations = isv_enroll_iterations

    self.m_gmm_isv_split = gmm_isv_split
    self.m_projected_toreplace = projected_toreplace
    self.m_projected_gmm = projected_gmm
    self.m_projected_isv = projected_isv
    self.m_projector_toreplace = projector_toreplace
    self.m_gmm_filename = gmm_filename
    self.m_isv_filename = isv_filename

  def _train_isv(self, data):
    """Train the ISV model given a dataset"""
    utils.info("  -> Training ISV enroller")
    self.m_isvbase = bob.machine.ISVBase(self.m_ubm, self.m_subspace_dimension_of_u)
    # train ISV model
    t = bob.trainer.ISVTrainer(self.m_isv_training_iterations, self.m_relevance_factor)
    t.rng = bob.core.random.mt19937(self.m_init_seed)
    t.train(self.m_isvbase, data)


  def _load_train_isv(self, train_features):
    utils.info("  -> Projecting training data")
    data = []
    for client_features in train_features:
      list = []
      for feature in client_features:
        # Initializes GMMStats object
        self.m_gmm_stats = bob.machine.GMMStats(self.m_ubm.dim_c, self.m_ubm.dim_d)
        list.append(UBMGMM.project(self, feature))
      data.append(list)

    self._train_isv(data)

  def train_projector(self, train_features, projector_file):
    """Train Projector and Enroller at the same time"""

    data1 = numpy.vstack([feature for client in train_features for feature in client])

    UBMGMM._train_projector_using_array(self, data1)
    # to save some memory, we might want to delete these data
    del data1

    # train ISV
    self._load_train_isv(train_features)

    # Save the ISV base AND the UBM into the same file
    self._save_projector(projector_file)


  def _save_projector_together(self, projector_file):
    """Save the GMM and the ISV model in the same HDF5 file"""
    hdf5file = bob.io.HDF5File(projector_file, "w")
    hdf5file.create_group('Projector')
    hdf5file.cd('Projector')
    self.m_ubm.save(hdf5file)

    hdf5file.cd('/')
    hdf5file.create_group('Enroller')
    hdf5file.cd('Enroller')
    self.m_isvbase.save(hdf5file)


  def _resolve_gmm_filename(self, projector_file):
    return projector_file.replace(self.m_projector_toreplace, self.m_gmm_filename)

  def _resolve_isv_filename(self, projector_file):
    return projector_file.replace(self.m_projector_toreplace, self.m_isv_filename)

  def _resolve_projected_gmm(self, projected_file):
    return projected_file.replace(self.m_projected_toreplace, self.m_projected_gmm)

  def _resolve_projected_isv(self, projected_file):
    return projected_file.replace(self.m_projected_toreplace, self.m_projected_isv)


  def _save_projector_gmm_resolved(self, gmm_filename):
    self.m_ubm.save(bob.io.HDF5File(gmm_filename, "w"))

  def _save_projector_gmm(self, projector_file):
    gmm_filename = self._resolve_gmm_filename(projector_file)
    self._save_projector_gmm_resolved(gmm_filename)

  def _save_projector_isv_resolved(self, isv_filename):
    self.m_isvbase.save(bob.io.HDF5File(isv_filename, "w"))

  def _save_projector_isv(self, projector_file):
    isv_filename = self._resolve_isv_filename(projector_file)
    self._save_projector_isv_resolved(isv_filename)

  def _save_projector(self, projector_file):
    """Save the GMM and the ISV model"""
    if not self.m_gmm_isv_split:
      self._save_projector_together(projector_file)
    else:
      self._save_projector_gmm(projector_file)
      self._save_projector_isv(projector_file)


  # Here, we just need to load the UBM from the projector file.
  def _load_projector_gmm_resolved(self, gmm_filename):
    self.m_ubm = bob.machine.GMMMachine(bob.io.HDF5File(gmm_filename))
    self.m_ubm.set_variance_thresholds(self.m_variance_threshold)
    # Initializes GMMStats object
    self.m_gmm_stats = bob.machine.GMMStats(self.m_ubm.dim_c, self.m_ubm.dim_d)

  def _load_projector_gmm(self, projector_file):
    gmm_filename = self._resolve_gmm_filename(projector_file)
    self._load_projector_gmm_resolved(gmm_filename)

  def _load_projector_isv_resolved(self, isv_filename):
    self.m_isvbase = bob.machine.ISVBase(bob.io.HDF5File(isv_filename))
    # add UBM model from base class
    self.m_isvbase.ubm = self.m_ubm

  def _load_projector_isv(self, projector_file):
    isv_filename = self._resolve_isv_filename(projector_file)
    self._load_projector_isv_resolved(isv_filename)

  def _load_projector_together(self, projector_file):
    """Load the GMM and the ISV model from the same HDF5 file"""
    hdf5file = bob.io.HDF5File(projector_file)

    # Load Projector
    hdf5file.cd('/Projector')
    # read UBM
    self.m_ubm = bob.machine.GMMMachine(hdf5file)
    self.m_ubm.set_variance_thresholds(self.m_variance_threshold)
    # Initializes GMMStats object
    self.m_gmm_stats = bob.machine.GMMStats(self.m_ubm.dim_c, self.m_ubm.dim_d)

    # Load Enroller
    hdf5file.cd('/Enroller')
    self.m_isvbase = bob.machine.ISVBase(hdf5file)
    # add UBM model from base class
    self.m_isvbase.ubm = self.m_ubm

  def load_projector(self, projector_file):
    """Reads the UBM model from file"""

    if not self.m_gmm_isv_split:
      self._load_projector_together(projector_file)
    else:
      self._load_projector_gmm(projector_file)
      self._load_projector_isv(projector_file)

    self.m_machine = bob.machine.ISVMachine(self.m_isvbase)
    self.m_trainer = bob.trainer.ISVTrainer(self.m_isv_training_iterations, self.m_relevance_factor)
    self.m_trainer.rng = bob.core.random.mt19937(self.m_init_seed)


  #######################################################
  ################ ISV training #########################
  def _project_gmm(self, feature_array):
    return UBMGMM.project(self,feature_array)

  def _project_isv(self, projected_ubm):
    projected_isv = numpy.ndarray(shape=(self.m_ubm.dim_c*self.m_ubm.dim_d,), dtype=numpy.float64)
    model = bob.machine.ISVMachine(self.m_isvbase)
    model.estimate_ux(projected_ubm, projected_isv)
    return projected_isv

  def project(self, feature_array):
    """Computes GMM statistics against a UBM, then corresponding Ux vector"""
    projected_ubm = self._project_gmm(feature_array)
    projected_isv = self._project_isv(projected_ubm)
    return [projected_ubm, projected_isv]

  #######################################################
  ################## ISV model enroll ####################

  def _save_feature_together(self, gmmstats, Ux, feature_file):
    hdf5file = bob.io.HDF5File(feature_file, "w")
    hdf5file.create_group('gmmstats')
    hdf5file.cd('gmmstats')
    gmmstats.save(hdf5file)
    hdf5file.cd('/')
    hdf5file.set('Ux', Ux)

  def _save_feature_gmm(self, data, feature_file):
    feature_file_gmm = self._resolve_projected_gmm(feature_file)
    data.save(bob.io.HDF5File(str(feature_file_gmm), 'w'))

  def _save_feature_isv(self, data, feature_file):
    feature_file_isv = self._resolve_projected_isv(feature_file)
    bob.io.save(data, str(feature_file_isv))

  def save_feature(self, data, feature_file):
    gmmstats = data[0]
    Ux = data[1]
    if not self.m_gmm_isv_split:
      self._save_feature_together(gmmstats, Ux, feature_file)
    else:
      self._save_feature_gmm(gmmstats, feature_file)
      self._save_feature_isv(Ux, feature_file)

  def read_feature(self, feature_file):
    """Read the type of features that we require, namely GMMStats"""
    if not self.m_gmm_isv_split:
      hdf5file = bob.io.HDF5File(feature_file)
      hdf5file.cd('gmmstats')
      gmmstats = bob.machine.GMMStats(hdf5file)
    else:
      feature_file_gmm = self._resolve_projected_gmm(feature_file)
      gmmstats = bob.machine.GMMStats(bob.io.HDF5File(str(feature_file_gmm)))
    return gmmstats


  def enroll(self, enroll_features):
    """Performs ISV enrollment"""
    self.m_trainer.enrol(self.m_machine, enroll_features, self.m_isv_enroll_iterations)
    # return the resulting gmm
    return self.m_machine


  ######################################################
  ################ Feature comparison ##################
  def read_model(self, model_file):
    """Reads the ISV Machine that holds the model"""
    machine = bob.machine.ISVMachine(bob.io.HDF5File(model_file))
    machine.isv_base = self.m_isvbase
    return machine

  def read_probe(self, probe_file):
    """Read the type of features that we require, namely GMMStats"""
    if self.m_gmm_isv_split:
      probe_file_gmm = self._resolve_projected_gmm(probe_file)
      gmmstats = bob.machine.GMMStats(bob.io.HDF5File(str(probe_file_gmm)))
      probe_file_isv = self._resolve_projected_isv(probe_file)
      Ux = bob.io.load(str(probe_file_isv))
    else:
      hdf5file = bob.io.HDF5File(probe_file)
      hdf5file.cd('gmmstats')
      gmmstats = bob.machine.GMMStats(hdf5file)
      hdf5file.cd('/')
      Ux = hdf5file.read('Ux')
    return [gmmstats, Ux]

  def score(self, model, probe):
    """Computes the score for the given model and the given probe."""
    gmmstats = probe[0]
    Ux = probe[1]
    return model.forward_ux(gmmstats, Ux)

  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and several given probe files."""
    # create GMM statistics from first probe statistics
    gmmstats_acc = bob.machine.GMMStats(probes[0][0])
    # add all other probe statistics
    for i in range(1,len(probes)):
      gmmstats_acc += probes[i][0]
    # compute ISV score with the accumulated statistics
    projected_isv_acc = numpy.ndarray(shape=(self.m_ubm.dim_c*self.m_ubm.dim_d,), dtype=numpy.float64)
    model.estimate_ux(gmmstats_acc, projected_isv_acc)
    return model.forward_ux(gmmstats_acc, projected_isv_acc)






# Parent classes:
# - Warning: This class uses multiple inheritance! (Note: Python's resolution rule is: depth-first, left-to-right)
# - ISV extends UBMGMM by providing some additional methods for training the session variability subspace, etc.
# - UBMGMMVideo extends UBMGMM to support UBM training/enrollment/testing with video.FrameContainers
#
# Here we extend the parent classes by overriding methods:
# -- read_feature --> overridden (use UBMGMMVideo's, to read a video.FrameContainer)
# -- train_projector --> overridden (use UBMGMMVideo's)
# -- train_enroller --> overridden (based on ISV's, but projects only selected frames)
# -- project --> overridden (use UBMGMMVideo's)
# -- enroll --> overridden (based on ISV, but first projects only selected frames)
# -- read_model --> inherited from ISV (because it's inherited first)
# -- read_probe --> inherited from ISV (because it's inherited first)
# -- score --> inherited from ISV (because it's inherited first)

class ISVVideo (ISV, UBMGMMVideo):
  """Tool chain for video-to-video face recognition using inter-session variability modelling (ISV)."""

  def __init__(
      self,
      frame_selector_for_projector_training,
      frame_selector_for_projection,
      frame_selector_for_enroll,
       **kwargs
  ):

    # call only one base class constructor...
    ISV.__init__(self, **kwargs)

    # call tool constructor to overwrite what was set before
    Tool.__init__(
        self,
        performs_projection = True,
        use_projected_features_for_enrollment = False,
        requires_enroller_training = True
    )

    self.m_frame_selector_for_projector_training = frame_selector_for_projector_training
    self.m_frame_selector_for_projection = frame_selector_for_projection
    self.m_frame_selector_for_enroll = frame_selector_for_enroll

    utils.warn("In its current version, this class has not been tested. Use it with care!")


  # Overrides ISV.train_enroller
  def train_enroller(self, train_features, enroller_file):
    utils.debug(" .... ISVVideo.train_enroller")
    ########## (same as ISV.train_enroller)
    # create a ISVBase with the UBM from the base class
    self.m_isvbase = bob.machine.ISVBase(self.m_ubm, self.m_subspace_dimension_of_u)

    ########## calculate GMM stats from video.FrameContainers, using frame_selector_for_train_enroller
    gmm_stats = []
    for client_features in train_features: # loop over clients
      gmm_stats_client = []
      for frame_container in client_features: # loop over videos of client k
        this_gmm_stats = UBMGMMVideo.project(self, frame_container, self.m_frame_selector_for_enroller_training)
        gmm_stats_client.append(this_gmm_stats)
      gmm_stats.append(gmm_stats_client)

    utils.debug(" .... got gmm_stats for " + str(len(gmm_stats)) + " clients")

    ########## (same as ISV.train_enroller)
    t = bob.trainer.ISVTrainer(self.m_isv_training_iterations, self.m_relevance_factor)
    t.rng = bob.core.random.mt19937(self.m_init_seed)
    t.train(self.m_isvbase, gmm_stats)

    # Save the ISV base AND the UBM into the same file
    self.m_isvbase.save(bob.io.HDF5File(enroller_file, "w"))

  def enroll(self, frame_containers):
    utils.debug(" .... ISVVideo.enroll")
    enroll_features = []
    for frame_container in frame_containers:
      this_enroll_features = UBMGMMVideo.project(self, frame_container, self.m_frame_selector_for_enroll)
      enroll_features.append(this_enroll_features)
    utils.debug(" .... got " + str(len(enroll_features)) + " enroll_features")

    ########## (same as ISV.enroll)
    self.m_trainer.enroll(self.m_machine, enroll_features, self.m_isv_enroll_iterations)
    return self.m_machine

  def read_feature(self, feature_file):
    return UBMGMMVideo.read_feature(self,str(feature_file))

  def project(self, frame_container):
    """Computes GMM statistics against a UBM, given an input video.FrameContainer"""
    return UBMGMMVideo.project(self,frame_container)

  def train_projector(self, train_files, projector_file):
    """Computes the Universal Background Model from the training ("world") data"""
    return UBMGMMVideo.train_projector(self,train_files, projector_file)

