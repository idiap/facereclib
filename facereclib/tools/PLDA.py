#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy
import types


class PLDATool:
  """Tool chain for computing Unified Background Models and Gaussian Mixture Models of the features"""

  def __init__(self, setup):
    """Initializes the local UBM-GMM tool chain with the given file selector object"""
    self.m_config = setup

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


  def train_projector(self, training_features, projector_file):
    """Generates the PLDA base model from a list of Arraysets (one per identity),
       and a set of training parameters.
       Returns the trained PLDABaseMachine."""
       
    # read data
    data = self.__load_data_by_client__(training_features)
    
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
    self.m_plda_base_machine.save(bob.io.HDF5File(str(projector_file), "w"))


  def load_projector(self, projector_file):
    """Reads the UBM model from file"""
    # read UBM
    self.m_plda_base_machine = bob.machine.PLDABaseMachine(bob.io.HDF5File(projector_file))
    self.m_plda_machine = bob.machine.PLDAMachine(self.m_plda_base_machine)
    self.m_plda_trainer = bob.trainer.PLDATrainer(self.m_plda_machine)

  def enrol(self, enrol_features):
    """Enrols the model by computing an average of the given input vectors"""
    self.m_plda_trainer.enrol(bob.io.Arrayset(enrol_features))
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
    return model.forward(probe)

    

