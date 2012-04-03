#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy
import types


class UBMGMMTool:
  """Tool chain for computing Unified Background Models and Gaussian Mixture Models of the features"""

  
  def __init__(self, setup):
    """Initializes the local UBM-GMM tool chain with the given file selector object"""
    self.m_config = setup
    self.m_ubm = None
    if hasattr(self.m_config, 'scoring_function'):
      self.m_scoring_function = self.m_config.scoring_function
    
    self.use_unprojected_features_for_model_enrol = True

  #######################################################
  ################ UBM training #########################
  def __normalize_std_arrayset__(self, arrayset):
    """Applies a unit variance normalization to an arrayset"""
    # Loads the data in RAM
    arrayset.load()

    # Initializes variables
    length = arrayset.shape[0]
    n_samples = len(arrayset)
    mean = numpy.ndarray((length,), 'float64')
    std = numpy.ndarray((length,), 'float64')

    mean.fill(0)
    std.fill(0)

    # Computes mean and variance
    for array in arrayset:
      x = array.astype('float64')
      mean += x
      std += (x ** 2)

    mean /= n_samples
    std /= n_samples
    std -= (mean ** 2)
    std = std ** 0.5 # sqrt(std)

    ar_std = bob.io.Arrayset()
    for array in arrayset:
      ar_std.append(array.astype('float64') / std)

    return (ar_std,std)


  def __multiply_vectors_by_factors__(self, matrix, vector):
    """Used to unnormalise some data"""
    for i in range(0, matrix.shape[0]):
      for j in range(0, matrix.shape[1]):
        matrix[i, j] *= vector[j]

  
  #######################################################
  ################ UBM training #########################
  def train_projector(self, train_files, projector_file):
    """Computes the Unified Background Model from the training ("world") data"""

    print "Training UBM model with %d features" % len(train_files)
    
    # Loads the data into an Arrayset
    ar = bob.io.Arrayset()
    for k in sorted(train_files.keys()):
      ar.extend(bob.io.load(str(train_files[k])))

    # Computes input size
    input_size = ar.shape[0]

    # Normalizes the Arrayset if required
    if not self.m_config.norm_KMeans:
      normalized_ar = ar
    else:
      (normalized_ar,std_ar) = self.__normalize_std_arrayset__(ar)


    # Creates the machines (KMeans and GMM)
    kmeans = bob.machine.KMeansMachine(self.m_config.n_gaussians, input_size)
    self.m_ubm = bob.machine.GMMMachine(self.m_config.n_gaussians, input_size)

    # Creates the KMeansTrainer
    kmeans_trainer = bob.trainer.KMeansTrainer()
    kmeans_trainer.convergence_threshold = self.m_config.convergence_threshold
    kmeans_trainer.max_iterations = self.m_config.iterk

    # Trains using the KMeansTrainer
    kmeans_trainer.train(kmeans, normalized_ar)
    
    [variances, weights] = kmeans.get_variances_and_weights_for_each_cluster(normalized_ar)
    means = kmeans.means

    # Undoes the normalization
    if self.m_config.norm_KMeans:
      self.__multiply_vectors_by_factors__(means, std_ar)
      self.__multiply_vectors_by_factors__(variances, std_ar ** 2)

    # Initializes the GMM
    self.m_ubm.means = means
    self.m_ubm.variances = variances
    self.m_ubm.weights = weights
    self.m_ubm.set_variance_thresholds(self.m_config.variance_threshold)

    # Trains the GMM
    trainer = bob.trainer.ML_GMMTrainer(self.m_config.update_means, self.m_config.update_variances, self.m_config.update_weights)
    trainer.convergence_threshold = self.m_config.convergence_threshold
    trainer.max_iterations = self.m_config.iterg_train
    trainer.train(self.m_ubm, ar)

    # Saves the UBM to file
    print "Saving model to file", projector_file
    self.m_ubm.save(bob.io.HDF5File(projector_file, "w"))
    

  #######################################################
  ############## GMM training using UBM #################

  def load_projector(self, projector_file):
    """Reads the UBM model from file"""
    # read UBM
    self.m_ubm = bob.machine.GMMMachine(bob.io.HDF5File(projector_file))
    self.m_ubm.set_variance_thresholds(self.m_config.variance_threshold)
    # prepare MAP_GMM_Trainer
    if self.m_config.responsibilities_threshold == 0.:
      self.m_trainer = bob.trainer.MAP_GMMTrainer(self.m_config.relevance_factor, True, False, False)
    else:
      self.m_trainer = bob.trainer.MAP_GMMTrainer(self.m_config.relevance_factor, True, False, False, self.m_config.responsibilities_threshold)
    self.m_trainer.convergence_threshold = self.m_config.convergence_threshold
    self.m_trainer.max_iterations = self.m_config.iterg_enrol
    self.m_trainer.set_prior_gmm(self.m_ubm)
    
    # Initializes GMMStats object 
    self.m_gmm_stats = bob.machine.GMMStats(self.m_ubm.dim_c, self.m_ubm.dim_d)


  def project(self, feature):
    """Computes GMM statistics against a UBM"""
    
    feature = bob.io.Arrayset(feature)
    # Accumulates statistics
    self.m_gmm_stats.init()
    self.m_ubm.acc_statistics(feature, self.m_gmm_stats)

    # return the resulting statistics
    return self.m_gmm_stats


  def enrol(self, enrol_features):
    """Enrols a GMM using MAP adaptation"""
    
    # Loads the data into an Arrayset
    ar = bob.io.Arrayset()
    for feature in enrol_features:
      ar.extend(feature)

    # Creates a GMM from the UBM
    gmm = bob.machine.GMMMachine(self.m_ubm)
    gmm.set_variance_thresholds(self.m_config.variance_threshold)

    # Trains the GMM
    self.m_trainer.train(gmm, ar)

    # return the resulting gmm    
    return gmm

  
  ######################################################
  ################ Feature comparison ##################
  def read_model(self, model_file):
    return bob.machine.GMMMachine(bob.io.HDF5File(model_file))

  def read_probe(self, probe_file):
    """Read the type of features that we require, namely GMM_Stats"""
    return bob.machine.GMMStats(bob.io.HDF5File(probe_file))

  def score(self, model, probe):
    """Computes the score for the given model and the given probe using the scoring function from the config file"""
    return self.m_scoring_function([model], self.m_ubm, [probe])

