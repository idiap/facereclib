#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import bob
import numpy

from .Tool import Tool
from .. import utils


class PLDA (Tool):
  """Tool chain for computing PLDA (over PCA-dimensionality reduced) features"""

  def __init__(
      self,
      subspace_dimension_of_f, # Size of subspace F
      subspace_dimension_of_g, # Size of subspace G
      subspace_dimension_pca = None,  # if given, perform PCA on data and reduce the PCA subspace to the given dimension
      plda_training_iterations = 200, # Maximum number of iterations for the EM loop
      # TODO: refactor the remaining parameters!
      INIT_SEED = 5489, # seed for initializing
      INIT_F_METHOD = bob.trainer.PLDATrainer.BETWEEN_SCATTER,
      INIT_F_RATIO = 1,
      INIT_G_METHOD = bob.trainer.PLDATrainer.WITHIN_SCATTER,
      INIT_G_RATIO = 1,
      INIT_S_METHOD = bob.trainer.PLDATrainer.VARIANCE_DATA,
      INIT_S_RATIO = 1,
      multiple_probe_scoring = 'joint_likelihood'
  ):

    """Initializes the local (PCA-)PLDA tool chain with the given file selector object"""
    # call base class constructor and register that this class requires training for enrollment
    Tool.__init__(
        self,
        requires_enroller_training = True,

        subspace_dimension_of_f = subspace_dimension_of_f, # Size of subspace F
        subspace_dimension_of_g = subspace_dimension_of_g, # Size of subspace G
        subspace_dimension_pca = subspace_dimension_pca,  # if given, perform PCA on data and reduce the PCA subspace to the given dimension
        plda_training_iterations = plda_training_iterations, # Maximum number of iterations for the EM loop
        # TODO: refactor the remaining parameters!
        INIT_SEED = INIT_SEED, # seed for initializing
        INIT_F_METHOD = str(INIT_F_METHOD),
        INIT_F_RATIO = INIT_F_RATIO,
        INIT_G_METHOD = str(INIT_G_METHOD),
        INIT_G_RATIO = INIT_G_RATIO,
        INIT_S_METHOD =str(INIT_S_METHOD),
        INIT_S_RATIO = INIT_S_RATIO,
        multiple_probe_scoring = multiple_probe_scoring,
        multiple_model_scoring = None
    )

    self.m_subspace_dimension_of_f = subspace_dimension_of_f
    self.m_subspace_dimension_of_g = subspace_dimension_of_g
    self.m_subspace_dimension_pca = subspace_dimension_pca
    self.m_plda_training_iterations = plda_training_iterations
    self.m_score_set = {'joint_likelihood': 'joint_likelihood', 'average':numpy.average, 'min':min, 'max':max}[multiple_probe_scoring]

    # TODO: refactor
    self.m_init = (INIT_SEED, INIT_F_METHOD, INIT_F_RATIO, INIT_G_METHOD, INIT_G_RATIO, INIT_S_METHOD, INIT_S_RATIO)


  def __train_pca__(self, training_set):
    """Trains and returns a LinearMachine that is trained using PCA"""
    data_list = []
    for client in training_set:
      for feature in client:
        # Appends in the array
        data_list.append(feature)
    data = numpy.vstack(data_list)

    utils.info("  -> Training LinearMachine using PCA ")
    t = bob.trainer.PCATrainer()
    machine, __eig_vals = t.train(data)
    # limit number of pcs
    machine.resize(machine.shape[0], self.m_subspace_dimension_pca)
    return machine

  def __perform_pca_client__(self, machine, client):
    """Perform PCA on an array"""
    client_data_list = []
    for feature in client:
      # project data
      projected_feature = numpy.ndarray(machine.shape[1], numpy.float64)
      machine(feature, projected_feature)
      # add data in new array
      client_data_list.append(projected_feature)
    client_data = numpy.vstack(client_data_list)
    return client_data

  def __perform_pca__(self, machine, training_set):
    """Perform PCA on data"""
    data = []
    for client in training_set:
      client_data = self.__perform_pca_client__(machine, client)
      data.append(client_data)
    return data


  def train_enroller(self, training_features, projector_file):
    """Generates the PLDA base model from a list of arrays (one per identity),
       and a set of training parameters. If PCA is requested, it is trained on the same data.
       Both the trained PLDABase and the PCA machine are written."""


    # train PCA and perform PCA on training data
    if self.m_subspace_dimension_pca is not None:
      self.m_pca_machine = self.__train_pca__(training_features)
      training_features = self.__perform_pca__(self.m_pca_machine, training_features)

    input_dimension = training_features[0].shape[1]

    utils.info("  -> Training PLDA base machine")
    # create trainer
    t = bob.trainer.PLDATrainer(self.m_plda_training_iterations)

    #t.rng.seed = self.m_init[0]
    print self.m_init[0]
    t.rng = bob.core.random.mt19937(self.m_init[0])
    t.init_f_method = self.m_init[1]
    t.init_f_ratio = self.m_init[2]
    t.init_g_method = self.m_init[3]
    t.init_g_ratio = self.m_init[4]
    t.init_sigma_method = self.m_init[5]
    t.init_sigma_ratio = self.m_init[6]

    # train machine
    self.m_plda_base = bob.machine.PLDABase(input_dimension, self.m_subspace_dimension_of_f, self.m_subspace_dimension_of_g)
    t.train(self.m_plda_base, training_features)

    # write machines to file
    proj_hdf5file = bob.io.HDF5File(str(projector_file), "w")
    if self.m_subspace_dimension_pca is not None:
      proj_hdf5file.create_group('/pca')
      proj_hdf5file.cd('/pca')
      self.m_pca_machine.save(proj_hdf5file)
    proj_hdf5file.create_group('/plda')
    proj_hdf5file.cd('/plda')
    self.m_plda_base.save(proj_hdf5file)


  def load_enroller(self, projector_file):
    """Reads the PCA projection matrix and the PLDA model from file"""
    # read UBM
    proj_hdf5file = bob.io.HDF5File(projector_file)
    if self.m_subspace_dimension_pca is not None:
      proj_hdf5file.cd('/pca')
      self.m_pca_machine = bob.machine.LinearMachine(proj_hdf5file)
    proj_hdf5file.cd('/plda')
    self.m_plda_base = bob.machine.PLDABase(proj_hdf5file)
    #self.m_plda_base = bob.machine.PLDABase(bob.io.HDF5File(projector_file))
    self.m_plda_machine = bob.machine.PLDAMachine(self.m_plda_base)
    self.m_plda_trainer = bob.trainer.PLDATrainer()

  def enroll(self, enroll_features):
    """Enrolls the model by computing an average of the given input vectors"""
    if self.m_subspace_dimension_pca is not None:
      enroll_features_projected = self.__perform_pca_client__(self.m_pca_machine, enroll_features)
      self.m_plda_trainer.enrol(self.m_plda_machine,enroll_features_projected)
    else:
      self.m_plda_trainer.enrol(self.m_plda_machine,enroll_features)
    return self.m_plda_machine

  def read_model(self, model_file):
    """Reads the model, which in this case is a PLDA-Machine"""
    # read machine and attach base machine
    plda_machine = bob.machine.PLDAMachine(bob.io.HDF5File(model_file), self.m_plda_base)
    return plda_machine

  def score(self, model, probe):
    """Computes the PLDA score for the given model and probe"""
    return self.score_for_multiple_probes(model, [probe])

  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and several given probe files.
    In this base class implementation, it computes the scores for each probe file using the 'score' method,
    and fuses the scores using the fusion method specified in the constructor of this class."""
    n_probes = len(probes)
    if self.m_subspace_dimension_pca is not None:
      # project probe
      probe_ = numpy.ndarray((n_probes, self.m_pca_machine.shape[1]), numpy.float64)
      projected_probe = numpy.ndarray(self.m_pca_machine.shape[1], numpy.float64)
      for i in range(n_probes):
        self.m_pca_machine(probes[i], projected_probe)
        probe_[i,:] = projected_probe
      # forward
      if self.m_score_set == 'joint_likelihood':
        return model.forward(probe_)
      else:
        scores = [model.forward(probe_[i,:]) for i in range(n_probes)]
        return self.m_score_set(scores)
    else:
      # just forward
      if self.m_score_set == 'joint_likelihood':
        probe_ = numpy.ndarray((n_probes, self.m_pca_machine.shape[1]), numpy.float64)
        for i in range(n_probes):
          probe_[i,:] = probes[i]
        return model.forward(probe_)
      # forward
      else:
        scores = [model.forward(probes[i]) for i in range(n_probes)]
        return self.m_score_set(scores)
