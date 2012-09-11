#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import bob
import numpy
import types


class PCAPLDATool:
  """Tool chain for computing PLDA over PCA-dimensionality reduced features"""

  def __init__(self, setup):
    """Initializes the local PCA-PLDA tool chain with the given file selector object"""
    self.m_config = setup
    self.m_pca_subpace_size = setup.pca_subspace if hasattr(setup, 'pca_subspace') else None
    self.m_pca_machine = None

    # overwrite the training image list generation from the file selector
    # since PLDA needs training data to be split up into models
    self.use_training_features_sorted_by_identity = True


  def __load_data_by_client__(self, training_files):
    """Loads the data (arrays) from a list of list of filenames,
    one list for each client, and put them in a list of Arraysets."""

    # Initializes an arrayset for the data
    data = []
    for client in sorted(training_files.keys()):
      # Arrayset for this client
      client_data = bob.io.Arrayset()
      client_files = training_files[client]
      for k in sorted(client_files.keys()):
        # Loads the file
        feature = bob.io.load( str(client_files[k]) )
        # Appends in the arrayset
        client_data.append(feature)
      data.append(client_data)
    # Returns the list of Arraysets
    return data

  def __train_pca__(self, training_set):
    """Trains and returns a LinearMachine that is trained using PCA"""
    data = bob.io.Arrayset()
    for client in training_set:
      for feature in client:
        # Appends in the arrayset
        data.append(feature)

    print "Training LinearMachine using PCA (SVD)"
    t = bob.trainer.SVDPCATrainer()
    machine, __eig_vals = t.train(data)
    # limit number of pcs
    machine.resize(machine.shape[0], self.m_pca_subpace_size)
    return machine

  def __perform_pca_client__(self, machine, client):
    """Perform PCA on an Arrayset"""
    client_data = bob.io.Arrayset()
    for feature in client:
      # project data
      projected_feature = numpy.ndarray(machine.shape[1], numpy.float64)
      machine(feature, projected_feature)
      # add data in new arrayset
      client_data.append(projected_feature)
    return client_data

  def __perform_pca__(self, machine, training_set):
    """Perform PCA on data"""
    data = []
    for client in training_set:
      client_data = self.__perform_pca_client__(machine, client)
      data.append(client_data)
    return data


  def train_projector(self, training_features, projector_file):
    """Generates the PLDA base model from a list of Arraysets (one per identity),
       and a set of training parameters.
       Returns the trained PLDABaseMachine."""

    # read data
    data = self.__load_data_by_client__(training_features)

    if self.m_pca_subpace_size:
      self.m_pca_machine = self.__train_pca__(data)
      data = self.__perform_pca__(self.m_pca_machine, data)

    # create trainer
    t = bob.trainer.PLDABaseTrainer(self.m_config.nf, self.m_config.ng, self.m_config.acc, self.m_config.n_iter, False)
    t.seed = self.m_config.seed
    t.init_f_method = self.m_config.initFmethod
    t.init_f_ratio = self.m_config.initFratio
    t.init_g_method = self.m_config.initGmethod
    t.init_g_ratio = self.m_config.initGratio
    t.init_sigma_method = self.m_config.initSmethod
    t.init_sigma_ratio = self.m_config.initSratio

    # train machine
    self.m_plda_base_machine = bob.machine.PLDABaseMachine(self.m_config.n_inputs, self.m_config.nf, self.m_config.ng)
    t.train(self.m_plda_base_machine, data)

    # write machine to file
    proj_hdf5file = bob.io.HDF5File(str(projector_file), "w")
    if self.m_pca_subpace_size:
      proj_hdf5file.create_group('/pca')
      proj_hdf5file.cd('/pca')
      self.m_pca_machine.save(proj_hdf5file)
    proj_hdf5file.create_group('/plda')
    proj_hdf5file.cd('/plda')
    self.m_plda_base_machine.save(proj_hdf5file)
    #self.m_plda_base_machine.save(bob.io.HDF5File(str(projector_file), "w"))


  def load_projector(self, projector_file):
    """Reads the UBM model from file"""
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
    enroll_features_projected = self.__perform_pca_client__(self.m_pca_machine, enroll_features)
    self.m_plda_trainer.enrol(enroll_features_projected)
    #self.m_plda_trainer.enrol(bob.io.Arrayset(enrol_features))
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
    # Project probe
    projected_probe = numpy.ndarray(self.m_pca_machine.shape[1], numpy.float64)
    self.m_pca_machine(probe, projected_probe)
    return model.forward(projected_probe)

