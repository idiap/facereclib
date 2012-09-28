#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import bob
import numpy

from .. import utils


class PLDATool:
  """Tool chain for computing PLDA (over PCA-dimensionality reduced) features"""

  def __init__(self, setup):
    """Initializes the local (PCA-)PLDA tool chain with the given file selector object"""
    self.m_config = setup
    self.m_pca_subpace_size = setup.SUBSPACE_DIMENSION_PCA if hasattr(setup, 'SUBSPACE_DIMENSION_PCA') else None
    self.m_pca_machine = None


  def __load_data_by_client__(self, training_features):
    """Loads the data (arrays) from a list of list of filenames,
    one list for each client, and put them in a list of arraysets."""

    # Initializes an arrayset for the data
    data = []
    for client in sorted(training_features.keys()):
      # arrayset for this client
      client_features = training_features[client]
      client_data = numpy.vstack([client_features[k] for k in sorted(client_features.keys())])
      data.append(client_data)
    # Returns the list of arraysets
    return data

  def __train_pca__(self, training_set):
    """Trains and returns a LinearMachine that is trained using PCA"""
    data_list = []
    for client in training_set:
      for feature in client:
        # Appends in the arrayset
        data_list.append(feature)
    data = numpy.vstack(data_list)

    utils.info("  -> Training LinearMachine using PCA (SVD)")
    t = bob.trainer.SVDPCATrainer()
    machine, __eig_vals = t.train(data)
    # limit number of pcs
    machine.resize(machine.shape[0], self.m_pca_subpace_size)
    return machine

  def __perform_pca_client__(self, machine, client):
    """Perform PCA on an arrayset"""
    client_data_list = []
    for feature in client:
      # project data
      projected_feature = numpy.ndarray(machine.shape[1], numpy.float64)
      machine(feature, projected_feature)
      # add data in new arrayset
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
    """Generates the PLDA base model from a list of arraysets (one per identity),
       and a set of training parameters. If PCA is requested, it is trained on the same data.
       Both the trained PLDABaseMachine and the PCA machine are written."""

    # read data
    data = self.__load_data_by_client__(training_features)

    if self.m_pca_subpace_size:
      self.m_pca_machine = self.__train_pca__(data)
      data = self.__perform_pca__(self.m_pca_machine, data)

    input_dimension = data[0].shape[0]

    utils.info("  -> Training PLDA base machine")
    # create trainer
    t = bob.trainer.PLDABaseTrainer(
        self.m_config.SUBSPACE_DIMENSION_OF_F,
        self.m_config.SUBSPACE_DIMENSION_OF_G,
        self.m_config.PLDA_TRAINING_THRESHOLD,
        self.m_config.PLDA_TRAINING_ITERATIONS,
        False)

    t.seed = self.m_config.INIT_SEED
    t.init_f_method = self.m_config.INIT_F_METHOD
    t.init_f_ratio = self.m_config.INIT_F_RATIO
    t.init_g_method = self.m_config.INIT_G_METHOD
    t.init_g_ratio = self.m_config.INIT_G_RATIO
    t.init_sigma_method = self.m_config.INIT_S_METHOD
    t.init_sigma_ratio = self.m_config.INIT_S_RATIO

    # train machine
    self.m_plda_base_machine = bob.machine.PLDABaseMachine(input_dimension, self.m_config.SUBSPACE_DIMENSION_OF_F, self.m_config.SUBSPACE_DIMENSION_OF_G)
    t.train(self.m_plda_base_machine, data)

    # write machines to file
    proj_hdf5file = bob.io.HDF5File(str(projector_file), "w")
    if self.m_pca_subpace_size:
      proj_hdf5file.create_group('/pca')
      proj_hdf5file.cd('/pca')
      self.m_pca_machine.save(proj_hdf5file)
    proj_hdf5file.create_group('/plda')
    proj_hdf5file.cd('/plda')
    self.m_plda_base_machine.save(proj_hdf5file)


  def load_enroller(self, projector_file):
    """Reads the PCA projection matrix and the PLDA model from file"""
    # read UBM
    proj_hdf5file = bob.io.HDF5File(projector_file)
    if self.m_pca_subpace_size:
      proj_hdf5file.cd('/pca')
      self.m_pca_machine = bob.machine.LinearMachine(proj_hdf5file)
    proj_hdf5file.cd('/plda')
    self.m_plda_base_machine = bob.machine.PLDABaseMachine(proj_hdf5file)
    #self.m_plda_base_machine = bob.machine.PLDABaseMachine(bob.io.HDF5File(projector_file))
    self.m_plda_machine = bob.machine.PLDAMachine(self.m_plda_base_machine)
    self.m_plda_trainer = bob.trainer.PLDATrainer(self.m_plda_machine)

  def enroll(self, enroll_features):
    """Enrolls the model by computing an average of the given input vectors"""
    if self.m_pca_subpace_size:
      enroll_features_projected = self.__perform_pca_client__(self.m_pca_machine, enroll_features)
      self.m_plda_trainer.enrol(enroll_features_projected)
    else:
      self.m_plda_trainer.enrol(enroll_features)
    return self.m_plda_machine

  def read_model(self, model_file):
    """Reads the model, which in this case is a PLDA-Machine"""
    # read machine
    plda_machine = bob.machine.PLDAMachine(bob.io.HDF5File(model_file))
    # attach base machine
    plda_machine.plda_base = self.m_plda_base_machine
    return plda_machine

  def score(self, model, probe):
    """Computes the PLDA score for the given model and probe"""
    if self.m_pca_subpace_size:
      # project probe
      projected_probe = numpy.ndarray(self.m_pca_machine.shape[1], numpy.float64)
      self.m_pca_machine(probe, projected_probe)
      # forward
      return model.forward(projected_probe)
    else:
      # just forward
      return model.forward(probe)

