#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.core
import bob.io.base
import bob.learn.em

import numpy

from .Tool import Tool
from .. import utils

class UBMGMM (Tool):
  """Tool for computing Universal Background Models and Gaussian Mixture Models of the features"""

  def __init__(
      self,
      # parameters for the GMM
      number_of_gaussians,
      # parameters of UBM training
      k_means_training_iterations = 500, # Maximum number of iterations for K-Means
      gmm_training_iterations = 500,     # Maximum number of iterations for ML GMM Training
      training_threshold = 5e-4,         # Threshold to end the ML training
      variance_threshold = 5e-4,         # Minimum value that a variance can reach
      update_weights = True,
      update_means = True,
      update_variances = True,
      normalize_before_k_means = True,  # Normalize the input features before running K-Means
      # parameters of the GMM enrollment
      relevance_factor = 4,         # Relevance factor as described in Reynolds paper
      gmm_enroll_iterations = 1,    # Number of iterations for the enrollment phase
      responsibility_threshold = 0, # If set, the weight of a particular Gaussian will at least be greater than this threshold. In the case the real weight is lower, the prior mean value will be used to estimate the current mean and variance.
      INIT_SEED = 5489,
      # scoring
      scoring_function = bob.learn.em.linear_scoring
  ):
    """Initializes the local UBM-GMM tool chain with the given file selector object"""

    # call base class constructor and register that this tool performs projection
    Tool.__init__(
        self,
        performs_projection = True,
        use_projected_features_for_enrollment = False,

        number_of_gaussians = number_of_gaussians,
        k_means_training_iterations = k_means_training_iterations,
        gmm_training_iterations = gmm_training_iterations,
        training_threshold = training_threshold,
        variance_threshold = variance_threshold,
        update_weights = update_weights,
        update_means = update_means,
        update_variances = update_variances,
        normalize_before_k_means = normalize_before_k_means,
        relevance_factor = relevance_factor,
        gmm_enroll_iterations = gmm_enroll_iterations,
        responsibility_threshold = responsibility_threshold,
        INIT_SEED = INIT_SEED,
        scoring_function = str(scoring_function),

        multiple_model_scoring = None,
        multiple_probe_scoring = 'average'
    )

    # copy parameters
    self.m_gaussians = number_of_gaussians
    self.m_k_means_training_iterations = k_means_training_iterations
    self.m_gmm_training_iterations = gmm_training_iterations
    self.m_training_threshold = training_threshold
    self.m_variance_threshold = variance_threshold
    self.m_update_weights = update_weights
    self.m_update_means = update_means
    self.m_update_variances = update_variances
    self.m_normalize_before_k_means = normalize_before_k_means
    self.m_relevance_factor = relevance_factor
    self.m_gmm_enroll_iterations = gmm_enroll_iterations
    self.m_init_seed = INIT_SEED
    self.m_rng = bob.core.random.mt19937(self.m_init_seed)
    self.m_responsibility_threshold = responsibility_threshold
    self.m_scoring_function = scoring_function



  #######################################################
  ################ UBM training #########################
  def __normalize_std_array__(self, array):
    """Applies a unit variance normalization to an array"""

    # Initializes variables
    n_samples = array.shape[0]
    length = array.shape[1]
    mean = numpy.ndarray((length,), 'float64')
    std = numpy.ndarray((length,), 'float64')

    mean.fill(0)
    std.fill(0)

    # Computes mean and variance
    for k in range(n_samples):
      x = array[k,:].astype('float64')
      mean += x
      std += (x ** 2)

    mean /= n_samples
    std /= n_samples
    std -= (mean ** 2)
    std = std ** 0.5 # sqrt(std)

    ar_std_list = []
    for k in range(n_samples):
      ar_std_list.append(array[k,:].astype('float64') / std)
    ar_std = numpy.vstack(ar_std_list)

    return (ar_std,std)


  def __multiply_vectors_by_factors__(self, matrix, vector):
    """Used to unnormalize some data"""
    for i in range(0, matrix.shape[0]):
      for j in range(0, matrix.shape[1]):
        matrix[i, j] *= vector[j]


  #######################################################
  ################ UBM training #########################

  def _train_projector_using_array(self, array):

    utils.debug(" .... Training with %d feature vectors" % array.shape[0])

    # Computes input size
    input_size = array.shape[1]

    # Normalizes the array if required
    utils.debug(" .... Normalizing the array")
    if not self.m_normalize_before_k_means:
      normalized_array = array
    else:
      normalized_array, std_array = self.__normalize_std_array__(array)


    # Creates the machines (KMeans and GMM)
    utils.debug(" .... Creating machines")
    kmeans = bob.learn.em.KMeansMachine(self.m_gaussians, input_size)
    self.m_ubm = bob.learn.em.GMMMachine(self.m_gaussians, input_size)

    # Creates the KMeansTrainer
    kmeans_trainer = bob.learn.em.KMeansTrainer()

    # Trains using the KMeansTrainer
    utils.info("  -> Training K-Means")
    bob.learn.em.train(kmeans_trainer, kmeans, normalized_array, self.m_gmm_training_iterations, self.m_training_threshold, bob.core.random.mt19937(self.m_init_seed))

    variances, weights = kmeans.get_variances_and_weights_for_each_cluster(normalized_array)
    means = kmeans.means

    # Undoes the normalization
    utils.debug(" .... Undoing normalization")
    if self.m_normalize_before_k_means:
      self.__multiply_vectors_by_factors__(means, std_array)
      self.__multiply_vectors_by_factors__(variances, std_array ** 2)

    # Initializes the GMM
    self.m_ubm.means = means
    self.m_ubm.variances = variances
    self.m_ubm.weights = weights
    self.m_ubm.set_variance_thresholds(self.m_variance_threshold)

    # Trains the GMM
    utils.info("  -> Training GMM")
    trainer = bob.learn.em.ML_GMMTrainer(self.m_update_means, self.m_update_variances, self.m_update_weights)
    bob.learn.em.train(trainer, self.m_ubm, array, self.m_gmm_training_iterations, self.m_training_threshold, bob.core.random.mt19937(self.m_init_seed))


  def _save_projector(self, projector_file):
    """Save projector to file"""
    # Saves the UBM to file
    utils.debug(" .... Saving model to file '%s'" % projector_file)
    self.m_ubm.save(bob.io.base.HDF5File(projector_file, "w"))


  def train_projector(self, train_features, projector_file):
    """Computes the Universal Background Model from the training ("world") data"""

    utils.info("  -> Training UBM model with %d training files" % len(train_features))

    # Loads the data into an array
    array = numpy.vstack(train_features)

    self._train_projector_using_array(array)

    self._save_projector(projector_file)


  #######################################################
  ############## GMM training using UBM #################

  def load_ubm(self, ubm_file):
    hdf5file = bob.io.base.HDF5File(ubm_file)
    # read UBM
    self.m_ubm = bob.learn.em.GMMMachine(hdf5file)
    self.m_ubm.set_variance_thresholds(self.m_variance_threshold)
    # Initializes GMMStats object
    self.m_gmm_stats = bob.learn.em.GMMStats(self.m_ubm.shape[0], self.m_ubm.shape[1])


  def load_projector(self, projector_file):
    """Reads the UBM model from file"""
    # read UBM
    self.load_ubm(projector_file)
    # prepare MAP_GMM_Trainer
    kwargs = dict(mean_var_update_responsibilities_threshold=self.m_responsibility_threshold) if self.m_responsibility_threshold > 0. else dict()
    self.m_trainer = bob.learn.em.MAP_GMMTrainer(self.m_ubm, relevance_factor = self.m_relevance_factor, update_means = True, update_variances = False, **kwargs)
    self.m_rng = bob.core.random.mt19937(self.m_init_seed)


  def _project_using_array(self, array):
    utils.debug(" .... Projecting %d feature vectors" % array.shape[0])
    # Accumulates statistics
    self.m_gmm_stats.init()
    self.m_ubm.acc_statistics(array, self.m_gmm_stats)

    # return the resulting statistics
    return self.m_gmm_stats


  def project(self, feature_array):
    """Computes GMM statistics against a UBM, given an input 2D numpy.ndarray of feature vectors"""
    return self._project_using_array(feature_array)


  def read_gmm_stats(self, gmm_stats_file):
    """Reads GMM stats from file."""
    return bob.learn.em.GMMStats(bob.io.base.HDF5File(gmm_stats_file))

  def read_feature(self, feature_file):
    """Read the type of features that we require, namely GMM_Stats"""
    return self.read_gmm_stats(feature_file)


  def _enroll_using_array(self, array):
    utils.debug(" .... Enrolling with %d feature vectors" % array.shape[0])

    gmm = bob.learn.em.GMMMachine(self.m_ubm)
    gmm.set_variance_thresholds(self.m_variance_threshold)
    bob.learn.em.train(self.m_trainer, gmm, array, self.m_gmm_enroll_iterations, self.m_training_threshold, self.m_rng)
    return gmm

  def enroll(self, feature_arrays):
    """Enrolls a GMM using MAP adaptation, given a list of 2D numpy.ndarray's of feature vectors"""
    array = numpy.vstack([v for v in feature_arrays])
    # Use the array to train a GMM and return it
    return self._enroll_using_array(array)


  ######################################################
  ################ Feature comparison ##################
  def read_model(self, model_file):
    return bob.learn.em.GMMMachine(bob.io.base.HDF5File(model_file))

  def score(self, model, probe):
    """Computes the score for the given model and the given probe using the scoring function from the config file"""
    return self.m_scoring_function([model], self.m_ubm, [probe], [], frame_length_normalisation = True)[0][0]

  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and several given probe files."""
    utils.warn("Please verify that this function is correct")
    return self.m_probe_fusion_function(self.m_scoring_function([model], self.m_ubm, probes, [], frame_length_normalisation = True))





class UBMGMMRegular (UBMGMM):
  """Tool chain for computing Universal Background Models and Gaussian Mixture Models of the features"""

  def __init__(self, **kwargs):
    """Initializes the local UBM-GMM tool chain with the given file selector object"""
    utils.warn("This class must be checked. Please verify that I didn't do any mistake here. I had to rename 'train_projector' into a 'train_enroller'!")
    # initialize the UBMGMM base class
    UBMGMM.__init__(self, **kwargs)
    # register a different set of functions in the Tool base class
    Tool.__init__(self, requires_enroller_training = True)



  #######################################################
  ################ UBM training #########################

  def train_enroller(self, train_features, enroller_file):
    """Computes the Universal Background Model from the training ("world") data"""
    return self.train_projector(train_features, enroller_file)


  #######################################################
  ############## GMM training using UBM #################

  def load_enroller(self, enroller_file):
    """Reads the UBM model from file"""
    return self.load_projector(enroller_file)


  ######################################################
  ################ Feature comparison ##################
  def read_probe(self, probe_file):
    return utils.load(probe_file)


  def score(self, model, probe):
    """Computes the score for the given model and the given probe.
       The score are Log-Likelihood.
       Therefore, the log of the likelihood ratio is obtained by computing the following difference."""

    utils.warn("This class must be checked. Please verify that I didn't do any mistake here. For identical tests, this function gives a different score than the normal UBMGMM (see test_tools.py:test06a)")
    score = 0
    for i in range(probe.shape[0]):
      score += model.log_likelihood(probe[i,:]) - self.m_ubm.log_likelihood(probe[i,:])
    return score/probe.shape[0]
