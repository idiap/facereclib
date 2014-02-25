#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import bob
import numpy
import types

from .Tool import Tool
from .UBMGMM import UBMGMM
from .. import utils
from itertools import izip
from math import sqrt

class IVector (UBMGMM):
  """Tool for extracting I-Vectors"""

  def __init__(
      self,
      # IVector training
      subspace_dimension_of_t,       # T subspace dimension
      update_sigma = True,
      tv_training_iterations = 25,  # Number of EM iterations for the JFA training
      variance_threshold = 1e-5,
      # Parameters when splitting GMM and IVector files
      gmm_ivec_split = False,
      projected_toreplace = 'projected', # 'Magic' string in path that will be replaced by the GMM or IVector one
      projected_gmm = 'projected_gmm', # subdirectory for the projected gmm
      projected_ivec = 'projected_ivec', # subdirectory for the projected ivec
      projector_toreplace = 'Projector.hdf5', # 'Magic' string in path that will be replaced by the GMM or IVector one
      gmm_filename = 'gmm.hdf5', # filename for the GMM model
      ivec_filename = 'ivec.hdf5', # filename for the IVector model
      whitening_filename = 'whitening.hdf5', # filename for the IVector model
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
        gmm_ivec_split = gmm_ivec_split,
        projected_toreplace = projected_toreplace,
        projected_gmm = projected_gmm,
        projected_ivec = projected_ivec,
        projector_toreplace = projector_toreplace,
        gmm_filename = gmm_filename,
        ivec_filename = ivec_filename,
        whitening_filename = whitening_filename,

        multiple_model_scoring = None,
        multiple_probe_scoring = None,
        **kwargs
    )

    self.m_update_sigma = update_sigma
    self.m_subspace_dimension_of_t = subspace_dimension_of_t
    self.m_tv_training_iterations = tv_training_iterations
    self.m_variance_threshold = variance_threshold

    self.m_gmm_ivec_split = gmm_ivec_split
    self.m_projected_toreplace = projected_toreplace
    self.m_projected_gmm = projected_gmm
    self.m_projected_ivec = projected_ivec
    self.m_projector_toreplace = projector_toreplace
    self.m_gmm_filename = gmm_filename
    self.m_ivec_filename = ivec_filename
    self.m_whitening_filename = whitening_filename

  def _train_ivector(self, data):
    """Train the IVector model given a dataset"""
    utils.info("  -> Training IVector enroller")
    self.m_tv = bob.machine.IVectorMachine(self.m_ubm, self.m_subspace_dimension_of_t)
    self.m_tv.variance_threshold = self.m_variance_threshold

    # train IVector model
    t = bob.trainer.IVectorTrainer(update_sigma=self.m_update_sigma, max_iterations=self.m_tv_training_iterations)
    t.rng = bob.core.random.mt19937(self.m_init_seed)
    t.train(self.m_tv, data)

  def _load_train_ivector(self, train_features):
    utils.info("  -> Projecting training data")
    data = []
    for feature in train_features:
      # Initializes GMMStats object
      self.m_gmm_stats = bob.machine.GMMStats(self.m_ubm.dim_c, self.m_ubm.dim_d)
      data.append(UBMGMM.project(self, feature))

    self._train_ivector(data)
    return data

  def _train_whitening(self, training_features):
   # load GMM stats from training files
    ivectors_matrix = numpy.vstack(training_features)
    # create a Linear Machine     # Runs whitening (first method)
    self.m_whitening_machine = bob.machine.LinearMachine(ivectors_matrix.shape[1],ivectors_matrix.shape[1])
    # create the whitening trainer
    t = bob.trainer.WhiteningTrainer()

    t.train(self.m_whitening_machine, ivectors_matrix)
    # Save the whitening linear machine
#    print("Saving the whitening machine..")
#    self.m_whitening_machine.save(bob.io.HDF5File(whitening_enroller_file, "w"))


  def train_projector(self, train_features, projector_file):
    """Train Projector and Enroller at the same time"""

    data = numpy.vstack(train_features)

    UBMGMM._train_projector_using_array(self, data)
    # to save some memory, we might want to delete these data
    del data

    # train IVector
    training_gmms = self._load_train_ivector(train_features)

    # project training i-vectors
    whitening_train_data = [self._project_ivector(f) for f in training_gmms]
    self._train_whitening(whitening_train_data)

    # Save the IVector base AND the UBM into the same file
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
    self.m_tv.save(hdf5file)

    hdf5file.cd('/')
    hdf5file.create_group('Whitening')
    hdf5file.cd('Whitening')
    self.m_whitening_machine.save(hdf5file)


  def _resolve_gmm_filename(self, projector_file):
    return projector_file.replace(self.m_projector_toreplace, self.m_gmm_filename)

  def _resolve_ivector_filename(self, projector_file):
   return projector_file.replace(self.m_projector_toreplace, self.m_ivec_filename)

  def _resolve_whitening_filename(self, projector_file):
    return projector_file.replace(self.m_projector_toreplace, self.m_whitening_filename)

  def _resolve_projected_gmm(self, projected_file):
    return projected_file.replace(self.m_projected_toreplace, self.m_projected_gmm)

  def _resolve_projected_ivector(self, projected_file):
    return projected_file.replace(self.m_projected_toreplace, self.m_projected_ivec)


  def _save_projector_gmm_resolved(self, gmm_filename):
    self.m_ubm.save(bob.io.HDF5File(gmm_filename, "w"))

  def _save_projector_gmm(self, projector_file):
    gmm_filename = self._resolve_gmm_filename(projector_file)
    self._save_projector_gmm_resolved(gmm_filename)

  def _save_projector_ivector_resolved(self, ivec_filename):
    self.m_tv.save(bob.io.HDF5File(ivec_filename, "w"))

  def _save_projector_ivector(self, projector_file):
    ivec_filename = self._resolve_ivector_filename(projector_file)
    self._save_projector_ivector_resolved(ivec_filename)

  def _save_projector_whitening_resolved(self, whitening_filename):
    self.m_tv.save(bob.io.HDF5File(whitening_filename, "w"))

  def _save_projector_whitening(self, projector_file):
    whitening_filename = self._resolve_whitening_filename(projector_file)
    self._save_projector_ivector_resolved(whitening_filename)

  def _save_projector(self, projector_file):
    """Save the GMM and the IVector model"""
    if not self.m_gmm_ivec_split:
      self._save_projector_together(projector_file)
    else:
      self._save_projector_gmm(projector_file)
      self._save_projector_ivector(projector_file)
      self._save_projector_whitening(projector_file)


  # Here, we just need to load the UBM from the projector file.
  def _load_projector_gmm_resolved(self, gmm_filename):
    self.m_ubm = bob.machine.GMMMachine(bob.io.HDF5File(gmm_filename))
    self.m_ubm.set_variance_thresholds(self.m_variance_threshold)
    # Initializes GMMStats object
    self.m_gmm_stats = bob.machine.GMMStats(self.m_ubm.dim_c, self.m_ubm.dim_d)

  def _load_projector_gmm(self, projector_file):
    gmm_filename = self._resolve_gmm_filename(projector_file)
    self._load_projector_gmm_resolved(gmm_filename)

  def _load_projector_ivector_resolved(self, ivec_filename):
    self.m_tv = bob.machine.IVectorMachine(bob.io.HDF5File(ivec_filename))
    # add UBM model from base class
    self.m_tv.ubm = self.m_ubm

  def _load_projector_ivector(self, projector_file):
    ivec_filename = self._resolve_ivector_filename(projector_file)
    self._load_projector_ivector_resolved(ivec_filename)

  def _load_projector_whitening_resolved(self, whitening_filename):
    self.m_whitening_machine = bob.machine.LinearMachine(bob.io.HDF5File(whitening_filename))

  def _load_projector_whitening(self, projector_file):
    whitening_filename = self._resolve_whitening_filename(projector_file)
    self._load_projector_whitening_resolved(whitening_filename)

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
    self.m_tv = bob.machine.IVectorMachine(hdf5file)
    # add UBM model from base class
    self.m_tv.ubm = self.m_ubm

    # Load Whitening
    hdf5file.cd('/Whitening')
    self.m_whitening_machine = bob.machine.LinearMachine(hdf5file)

  def load_projector(self, projector_file):
    """Reads the UBM model from file"""

    if not self.m_gmm_ivec_split:
      self._load_projector_together(projector_file)
    else:
      self._load_projector_gmm(projector_file)
      self._load_projector_ivector(projector_file)
      self._load_projector_whitening(projector_file)


  #######################################################
  ################ ISV training #########################
  def _project_gmm(self, feature_array):
    return UBMGMM.project(self,feature_array)

  def _project_ivector(self, projected_ubm):
    return self.m_tv.forward(projected_ubm)

  def project(self, feature_array):
    """Computes GMM statistics against a UBM, then corresponding Ux vector"""
    projected_ubm = self._project_gmm(feature_array)
    projected_ivec = self._project_ivector(projected_ubm)
    return [projected_ubm, projected_ivec]

  #######################################################
  ################## ISV model enroll ####################

  def _save_feature_together(self, gmmstats, ivector, feature_file):
    hdf5file = bob.io.HDF5File(feature_file, "w")
    hdf5file.create_group('gmmstats')
    hdf5file.cd('gmmstats')
    gmmstats.save(hdf5file)
    hdf5file.cd('/')
    hdf5file.set('ivector', ivector)

  def _save_feature_gmm(self, data, feature_file):
    feature_file_gmm = self._resolve_projected_gmm(feature_file)
    data.save(bob.io.HDF5File(str(feature_file_gmm), 'w'))

  def _save_feature_ivector(self, data, feature_file):
    feature_file_ivec = self._resolve_projected_ivector(feature_file)
    bob.io.save(data, str(feature_file_ivec))

  def save_feature(self, data, feature_file):
    gmmstats = data[0]
    ivector = data[1]
    if not self.m_gmm_ivec_split:
      self._save_feature_together(gmmstats, ivector, feature_file)
    else:
      self._save_feature_gmm(gmmstats, feature_file)
      self._save_feature_ivector(ivector, feature_file)


  def _read_feature_gmm(self, feature_file):
    feature_file_gmm = self._resolve_projected_gmm(feature_file)
    return bob.machine.GMMStats(bob.io.HDF5File(str(feature_file_gmm)))

  def read_feature(self, feature_file):
    """Read the type of features that we require, namely GMMStats"""
    return bob.io.load(feature_file)


  ################ Whitening training #########################
  def train_whitening_projector(self, train_files, whitening_enroller_file):
    # load GMM stats from training files
    ivectors_matrix  = []
    for k in train_files:
      ivec = bob.io.load(str(k))
      ivectors_matrix.append(ivec)
    ivectors_matrix = numpy.vstack(ivectors_matrix)
    # create a Linear Machine     # Runs whitening (first method)
    self.m_whitening_machine = bob.machine.LinearMachine(ivectors_matrix.shape[1],ivectors_matrix.shape[1])
    # create the whitening trainer
    t = bob.trainer.WhiteningTrainer()

    t.train(self.m_whitening_machine, ivectors_matrix)
    # Save the whitening linear machine
#    print("Saving the whitening machine..")
    self.m_whitening_machine.save(bob.io.HDF5File(whitening_enroller_file, "w"))

  #######################################################
  ############## Whitening load model ##################
  def load_whitening_projector(self, whitening_projector_file):
    """Reads the whitening Enroller model from file"""
    # now, load the JFA base, if it is included in the file
    raise NotImplementedError("This function is outdated. Use _load_whitening_projector instead.")
    self.m_whitening_machine = bob.machine.LinearMachine(self.m_subspace_dimension_of_t,self.m_subspace_dimension_of_t)
    self.m_whitening_machine.load(bob.io.HDF5File(whitening_projector_file))

  #######################################################
  ####### whitening and length-normalize i-vectors ######
  def whitening_ivector(self, ivector):
    whitened_ivector = self.m_whitening_machine.forward(ivector)
    return whitened_ivector/numpy.linalg.norm(whitened_ivector)

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
    return bob.io.load(model_file)

  def read_probe(self, probe_file):
    """read probe file which is an i-vector"""
    return bob.io.load(probe_file)

  def score(self, model, probe):
    """Computes the score for the given model and the given probe."""
    a = model/numpy.linalg.norm(model)
#    print a
    b = probe/numpy.linalg.norm(probe)
#    print b
#    print a.shape, b.shape
    if len(a) != len(b):
        raise ValueError, "a and b must be same length"
    numerator = sum(tup[0] * tup[1] for tup in izip(a,b))
    return numerator


  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and several given probe files."""
    probes = numpy.vstack([numpy.mean(numpy.vstack(probes), axis=0)])
    return self.score(model,probes)

