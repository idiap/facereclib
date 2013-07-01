#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>


import sys, os, shutil
import argparse
import bob
import numpy

from . import ToolChainExecutor
from .. import toolchain, tools, utils


class ToolChainExecutorIVector (ToolChainExecutor.ToolChainExecutor):
  """Class that executes the ZT tool chain (locally or in the grid)."""

  def __init__(self, args):
    # call base class constructor
    ToolChainExecutor.ToolChainExecutor.__init__(self, args)

    if not isinstance(self.m_tool, tools.IVector):
      raise ValueError("This script is specifically designed to compute IVector tests. Please select an according tool.")

    self.m_tool.m_gmm_ivec_split = True

    if args.protocol:
      self.m_database.protocol = args.protocol

    self.m_configuration.normalized_directory = os.path.join(self.m_configuration.temp_directory, 'normalized_features')
    self.m_configuration.kmeans_file = os.path.join(self.m_configuration.temp_directory, 'k_means.hdf5')
    self.m_configuration.kmeans_intermediate_file = os.path.join(self.m_configuration.temp_directory, 'kmeans_temp', 'i_%05d', 'k_means.hdf5')
    self.m_configuration.kmeans_stats_file = os.path.join(self.m_configuration.temp_directory, 'kmeans_temp', 'i_%05d', 'stats_%05d-%05d.hdf5')
    self.m_tool.m_gmm_filename = os.path.join(self.m_configuration.temp_directory, 'gmm.hdf5')
    self.m_configuration.gmm_intermediate_file = os.path.join(self.m_configuration.temp_directory, 'gmm_temp', 'i_%05d', 'gmm.hdf5')
    self.m_configuration.gmm_stats_file = os.path.join(self.m_configuration.temp_directory, 'gmm_temp', 'i_%05d', 'stats_%05d-%05d.hdf5')
    self.m_configuration.ivector_intermediate_file = os.path.join(self.m_configuration.temp_directory, 'tv_temp', 'i_%05d', 'ivec.hdf5')
    self.m_configuration.ivector_stats_file = os.path.join(self.m_configuration.temp_directory, 'tv_temp', 'i_%05d', 'stats_%05d-%05d.hdf5')
    self.m_tool.m_ivec_filename = os.path.join(self.m_configuration.temp_directory, 'ivec.hdf5')
    self.m_tool.m_projected_toreplace = 'projected'
    self.m_tool.m_projected_gmm = 'projected_gmm'
    self.m_tool.m_projected_ivec = 'projected_ivec'
    self.m_tool.m_projector_toreplace = self.m_configuration.projector_file

    # specify the file selector to be used
    self.m_file_selector = toolchain.FileSelector(
        self.m_database,
        preprocessed_directory = self.m_configuration.preprocessed_directory,
        extractor_file = self.m_configuration.extractor_file,
        features_directory = self.m_configuration.features_directory,
        projector_file = self.m_configuration.projector_file,
        projected_directory = self.m_configuration.projected_directory,
        enroller_file = False,
        model_directories = False,
        score_directories = False,
        zt_score_directories = False
    )

    # create the tool chain to be used to actually perform the parts of the experiments
    self.m_tool_chain = toolchain.ToolChain(self.m_file_selector)


#######################################################################################
####################  Functions that will be executed in the grid  ####################
#######################################################################################
  def training_list(self):
    """Returns the list of feature files that is required for training"""
    features = self.m_file_selector.training_list('features', 'train_projector')
    if self.m_args.normalize_features:
      return [f.replace(self.m_configuration.features_directory, self.m_configuration.normalized_directory) for f in features]
    else:
      return features

  def feature_normalization(self, indices, force=False):
    """Normalizes the list of features to have zero mean and unit variance (parallel)"""
    training_list = self.m_file_selector.training_list('features', 'train_projector')
    normalized_list = self.training_list()

    utils.info("IVector training: normalizing features from range(%d, %d)" % indices)

    # iterate through the files and normalize the features
    for index in range(indices[0], indices[1]):
      feature = self.m_extractor.read_feature(str(training_list[index]))

      mean, std = self.m_tool.__normalize_std_array__(feature)

      if self.m_tool_chain.__check_file__(normalized_list[index], force):
        utils.debug("Skipping file '%s'" % normalized_list[index])
      else:
        utils.ensure_dir(os.path.dirname(normalized_list[index]))
        f = bob.io.HDF5File(str(normalized_list[index]), 'w')
        f.set('mean', mean)
        f.set('std', std)
        utils.debug("Saved normalized feature %s" %str(normalized_list[index]))


  def kmeans_initialize(self, force=False):
    """Initializes the K-Means training (non-parallel)."""
    output_file = self.m_configuration.kmeans_intermediate_file % 0

    if self.m_tool_chain.__check_file__(output_file, force, 1000):
      utils.info("IVector training: Skipping KMeans initialization since the file '%s' already exists" % output_file)
    else:
      # read data
      utils.info("IVector training: initializing kmeans")
      training_list = self.training_list()
      data = numpy.vstack([self.m_extractor.read_feature(str(training_list[index])) for index in utils.quasi_random_indices(len(training_list), self.m_args.limit_training_examples)])

      # Perform KMeans initialization
      kmeans_machine = bob.machine.KMeansMachine(self.m_tool.m_gaussians, data.shape[1])
      # Creates the KMeansTrainer and call the initialization procedure
      kmeans_trainer = bob.trainer.KMeansTrainer()
      kmeans_trainer.initialize(kmeans_machine, data)
      utils.ensure_dir(os.path.dirname(output_file))
      kmeans_machine.save(bob.io.HDF5File(output_file, 'w'))
      utils.info("IVector training: saved initial KMeans machine to '%s'" % output_file)


  def kmeans_estep(self, indices, force=False):
    """Performs a single E-step of the K-Means algorithm (parallel)"""
    stats_file = self.m_configuration.kmeans_stats_file % (self.m_args.iteration, indices[0], indices[1])

    if  self.m_tool_chain.__check_file__(stats_file, force, 1000):
      utils.info("IVector training: Skipping KMeans E-Step since the file '%s' already exists" % stats_file)
    else:
      training_list = self.training_list()
      machine_file = self.m_configuration.kmeans_intermediate_file % self.m_args.iteration
      kmeans_machine = bob.machine.KMeansMachine(bob.io.HDF5File(machine_file))

      utils.info("IVector training: KMeans E-Step from range(%d, %d)" % indices)

      # read data
      data = numpy.vstack([self.m_extractor.read_feature(str(training_list[index])) for index in range(indices[0], indices[1])])

      kmeans_trainer = bob.trainer.KMeansTrainer()
      t = bob.machine.KMeansMachine(self.m_tool.m_gaussians, data.shape[1]) # Temporary Kmeans machine required for trainer initialization
      kmeans_trainer.initialize(t, data)

      # Performs the E-step
      kmeans_trainer.e_step(kmeans_machine, data)

      # write results to file
      dist = numpy.array([kmeans_trainer.average_min_distance])
      nsamples = numpy.array([indices[1] - indices[0]], dtype=numpy.float64)

      utils.ensure_dir(os.path.dirname(stats_file))
      f = bob.io.HDF5File(stats_file, 'w')
      f.set('zeros', kmeans_trainer.zeroeth_order_statistics)
      f.set('first', kmeans_trainer.first_order_statistics)
      f.set('dist', dist * nsamples)
      f.set('nsamples', nsamples)
      utils.info("IVector training: Wrote Stats file '%s'" % stats_file)


  def read_stats(self, filename):
    """Reads accumulated K-Means statistics from file"""
    utils.debug("IVector training: Reading stats file '%s'" % filename)
    f = bob.io.HDF5File(filename)
    zeroeth  = f.read('zeros')
    first    = f.read('first')
    nsamples = f.read('nsamples')
    dist     = f.read('dist')
    return (zeroeth, first, nsamples, dist)


  def kmeans_mstep(self, counts, force=False):
    """Performs a single M-step of the K-Means algorithm (non-parallel)"""
    old_machine_file = self.m_configuration.kmeans_intermediate_file % self.m_args.iteration
    new_machine_file = self.m_configuration.kmeans_intermediate_file % (self.m_args.iteration + 1)

    if  self.m_tool_chain.__check_file__(new_machine_file, force, 1000):
      utils.info("IVector training: Skipping KMeans M-Step since the file '%s' already exists" % new_machine_file)
    else:
      # get the files from e-step
      training_list = self.training_list()

      # try if there is one file containing all data
      if os.path.exists(self.m_configuration.kmeans_stats_file % (self.m_args.iteration, 0, len(training_list))):
        stats_file = self.m_configuration.kmeans_stats_file % (self.m_args.iteration, 0, len(training_list))
        # load stats file
        zeroeth, first, nsamples, dist = self.read_stats(stats_file)
      else:
        # load several files
        job_ids = range(self.__generate_job_array__(training_list, counts)[1])
        job_indices = [(counts * job_id, min(counts * (job_id+1), len(training_list))) for job_id in job_ids]
        stats_files = [self.m_configuration.kmeans_stats_file % (self.m_args.iteration, indices[0], indices[1]) for indices in job_indices]

        # read all stats files
        zeroeth, first, nsamples, dist = self.read_stats(stats_files[0])
        for stats_file in stats_files[1:]:
          zeroeth_, first_, nsamples_, dist_ = self.read_stats(stats_file)
          zeroeth += zeroeth_
          first += first_
          nsamples += nsamples_
          dist += dist_

      # read some features (needed for computation, but not really required)
      data = numpy.array(self.m_extractor.read_feature(str(training_list[0])))

      # Creates the KMeansTrainer
      kmeans_trainer = bob.trainer.KMeansTrainer()
      # Creates the KMeansMachine
      kmeans_machine = bob.machine.KMeansMachine(bob.io.HDF5File(old_machine_file))
      kmeans_trainer.initialize(kmeans_machine, data)

      kmeans_trainer.zeroeth_order_statistics = zeroeth
      kmeans_trainer.first_order_statistics = first
      kmeans_trainer.average_min_distance = dist

      # Performs the M-step
      kmeans_trainer.m_step(kmeans_machine, data) # data is not used in M-step
      utils.info("IVector training: Performed M step %d with result %f" % (self.m_args.iteration, dist/nsamples))

      # Save the K-Means model
      utils.ensure_dir(os.path.dirname(new_machine_file))
      kmeans_machine.save(bob.io.HDF5File(new_machine_file, 'w'))
      shutil.copy(new_machine_file, self.m_configuration.kmeans_file)
      utils.info("IVector training: Wrote new KMeans machine '%s'" % new_machine_file)

    if self.m_args.clean_intermediate and self.m_args.iteration > 0:
      old_file = self.m_configuration.kmeans_intermediate_file % (self.m_args.iteration-1)
      utils.info("Removing old intermediate directory '%s'" % os.path.dirname(old_file))
      shutil.rmtree(os.path.dirname(old_file))


  def gmm_initialize(self, force=False):
    """Initializes the GMM calculation with the result of the K-Means algorithm (non-parallel).
    This might require a lot of memory."""
    output_file = self.m_configuration.gmm_intermediate_file % 0

    if self.m_tool_chain.__check_file__(output_file, force, 800):
      utils.info("IVector Training: Skipping GMM initialization since '%s' already exists" % output_file)
    else:
      training_list = self.training_list()
      utils.info("IVector Training: Initializing GMM")

      # load KMeans machine
      kmeans_machine = bob.machine.KMeansMachine(bob.io.HDF5File(self.m_configuration.kmeans_file))

      # read features
      data = numpy.vstack([self.m_extractor.read_feature(str(training_list[index])) for index in utils.quasi_random_indices(len(training_list), self.m_args.limit_training_examples)])

      # Create initial GMM Machine
      gmm_machine = bob.machine.GMMMachine(self.m_tool.m_gaussians, data.shape[1])

      [variances, weights] = kmeans_machine.get_variances_and_weights_for_each_cluster(data)

      # Initializes the GMM
      gmm_machine.means = kmeans_machine.means
      gmm_machine.variances = variances
      gmm_machine.weights = weights
      gmm_machine.set_variance_thresholds(self.m_tool.m_variance_threshold)

      utils.ensure_dir(os.path.dirname(output_file))
      gmm_machine.save(bob.io.HDF5File(os.path.join(output_file), 'w'))
      utils.info("IVector Training: Wrote GMM file '%s'" % output_file)


  def gmm_estep(self, indices, force=False):
    """Performs a single E-step of the GMM training (parallel)."""
    stats_file = self.m_configuration.gmm_stats_file % (self.m_args.iteration, indices[0], indices[1])

    if  self.m_tool_chain.__check_file__(stats_file, force, 1000):
      utils.info("IVector training: Skipping GMM E-Step since the file '%s' already exists" % stats_file)
    else:
      training_list = self.training_list()
      machine_file = self.m_configuration.gmm_intermediate_file % self.m_args.iteration
      gmm_machine = bob.machine.GMMMachine(bob.io.HDF5File(machine_file))

      utils.info("IVector training: GMM E-Step from range(%d, %d)" % indices)

      # read data
      data = numpy.vstack([self.m_extractor.read_feature(str(training_list[index])) for index in range(indices[0], indices[1])])

      gmm_trainer = bob.trainer.ML_GMMTrainer(self.m_tool.m_update_means, self.m_tool.m_update_variances, self.m_tool.m_update_weights)
      gmm_trainer.responsibilities_threshold = self.m_tool.m_responsibility_threshold
      gmm_trainer.initialize(gmm_machine, data)

      # Calls the E-step and extracts the GMM statistics
      gmm_trainer.e_step(gmm_machine, data)
      gmm_stats = gmm_trainer.gmm_statistics

      # Saves the GMM statistics to the file
      utils.ensure_dir(os.path.dirname(stats_file))
      gmm_stats.save(bob.io.HDF5File(stats_file, 'w'))
      utils.info("IVector training: Wrote GMM stats '%s'" % (stats_file))


  def gmm_mstep(self, counts, force=False):
    """Performs a single M-step of the GMM training (non-parallel)"""
    old_machine_file = self.m_configuration.gmm_intermediate_file % self.m_args.iteration
    new_machine_file = self.m_configuration.gmm_intermediate_file % (self.m_args.iteration + 1)

    if  self.m_tool_chain.__check_file__(new_machine_file, force, 1000):
      utils.info("IVector training: Skipping GMM M-Step since the file '%s' already exists" % new_machine_file)
    else:
      # get the files from e-step
      training_list = self.training_list()

      # try if there is one file containing all data
      if os.path.exists(self.m_configuration.gmm_stats_file % (self.m_args.iteration, 0, len(training_list))):
        stats_file = self.m_configuration.gmm_stats_file % (self.m_args.iteration, 0, len(training_list))
        # load stats file
        zeroeth, first, nsamples, dist = self.read_stats(stats_file)
      else:
        # load several files
        job_ids = range(self.__generate_job_array__(training_list, counts)[1])
        job_indices = [(counts * job_id, min(counts * (job_id+1), len(training_list))) for job_id in job_ids]
        stats_files = [self.m_configuration.gmm_stats_file % (self.m_args.iteration, indices[0], indices[1]) for indices in job_indices]

        # read all stats files
        gmm_stats = bob.machine.GMMStats(bob.io.HDF5File(stats_files[0]))
        for stats_file in stats_files[1:]:
          gmm_stats += bob.machine.GMMStats(bob.io.HDF5File(stats_file))

      # read some features (needed for computation, but not really required)
      data = numpy.array(self.m_extractor.read_feature(str(training_list[0])))

      # load the old gmm machine
      gmm_machine =  bob.machine.GMMMachine(bob.io.HDF5File(old_machine_file))
      # initialize the trainer
      gmm_trainer = bob.trainer.ML_GMMTrainer(self.m_tool.m_update_means, self.m_tool.m_update_variances, self.m_tool.m_update_weights)
      gmm_trainer.responsibilities_threshold = self.m_tool.m_responsibility_threshold
      gmm_trainer.initialize(gmm_machine, data)
      gmm_trainer.gmm_statistics = gmm_stats

      # Calls M-step
      gmm_trainer.m_step(gmm_machine, data)

      # Saves the GMM statistics to the file
      utils.ensure_dir(os.path.dirname(new_machine_file))
      gmm_machine.save(bob.io.HDF5File(new_machine_file, 'w'))
      import shutil
      shutil.copy(new_machine_file, self.m_tool.m_gmm_filename)

    if self.m_args.clean_intermediate and self.m_args.iteration > 0:
      old_file = self.m_configuration.gmm_intermediate_file % (self.m_args.iteration-1)
      utils.info("Removing old intermediate directory '%s'" % os.path.dirname(old_file))
      shutil.rmtree(os.path.dirname(old_file))


  def gmm_project(self, indices, force=False):
    """Performs GMM projection"""
    # read UBM into the IVector class
    self.m_tool._load_projector_gmm_resolved(self.m_tool.m_gmm_filename)
      
    feature_files = self.m_file_selector.feature_list()
    projected_files = self.m_file_selector.projected_list()

    # select a subset of indices to iterate
    if indices != None:
      index_range = range(indices[0], indices[1])
      utils.info("- Projection: splitting of index range %s" % str(indices))
    else:
      index_range = range(len(feature_files))

    #utils.ensure_dir(self.m_file_selector.projected_directory)
    utils.info("- Projection: projecting %d images from directory '%s' to directory '%s'" % (len(index_range), self.m_file_selector.features_directory, self.m_tool._resolve_projected_gmm(self.m_file_selector.projected_directory)))
    # extract the features
    for i in index_range:
      feature_file = feature_files[i]
      projected_file = projected_files[i]
      projected_file_resolved = self.m_tool._resolve_projected_gmm(projected_file)

      if not self.m_tool_chain.__check_file__(projected_file_resolved, force):
        # load feature
        feature = self.m_extractor.read_feature(str(feature_file))
        # project feature
        projected = tools.UBMGMM.project(self.m_tool, feature)
        # write it
        utils.ensure_dir(os.path.dirname(projected_file_resolved))
        self.m_tool._save_feature_gmm(projected, str(projected_file))


  def ivector_initialize(self, force=False):
    """Initializes the IVector training (non-parallel)."""
    output_file = self.m_configuration.ivector_intermediate_file % 0

    if self.m_tool_chain.__check_file__(output_file, force, 1000):
      utils.info("IVector training: Skipping IVector initialization since the file '%s' already exists" % output_file)
    else:
      # read data
      utils.info("IVector training: initializing ivector")
      data = []
      # Perform IVector initialization
      ubm = bob.machine.GMMMachine(bob.io.HDF5File(self.m_tool.m_gmm_filename))
      ivector_machine = bob.machine.IVectorMachine(ubm, self.m_tool.m_subspace_dimension_of_t)
      ivector_machine.variance_threshold = self.m_tool.m_variance_threshold
      # Creates the IVectorTrainer and call the initialization procedure
      ivector_trainer = bob.trainer.IVectorTrainer(update_sigma=self.m_tool.m_update_sigma, max_iterations=self.m_tool.m_tv_training_iterations)
      ivector_trainer.initialize(ivector_machine, data)
      utils.ensure_dir(os.path.dirname(output_file))
      ivector_machine.save(bob.io.HDF5File(output_file, 'w'))
      utils.info("IVector training: saved initial IVector machine to '%s'" % output_file)


  def ivector_estep(self, indices, force=False):
    """Performs a single E-step of the IVector algorithm (parallel)"""
    stats_file = self.m_configuration.ivector_stats_file % (self.m_args.iteration, indices[0], indices[1])

    if  self.m_tool_chain.__check_file__(stats_file, force, 1000):
      utils.info("IVector training: Skipping IVector E-Step since the file '%s' already exists" % stats_file)
    else:
      utils.info("IVector training: E-Step from range(%d, %d)" % indices)

      # Temporary machine used for initialization
      ubm = bob.machine.GMMMachine(bob.io.HDF5File(self.m_tool.m_gmm_filename))
      m = bob.machine.IVectorMachine(ubm, self.m_tool.m_subspace_dimension_of_t)
      m.variance_threshold = self.m_tool.m_variance_threshold
      # Load machine
      machine_file = self.m_configuration.ivector_intermediate_file % self.m_args.iteration
      ivector_machine = bob.machine.IVectorMachine(bob.io.HDF5File(machine_file))
      ivector_machine.ubm = ubm
  
      # Load data
      training_list = self.m_file_selector.training_list('projected', 'train_projector')
      data = [self.m_tool.read_feature(str(training_list[index])) for index in range(indices[0], indices[1])]

      # Creates the IVectorTrainer and call the initialization procedure
      ivector_trainer = bob.trainer.IVectorTrainer(update_sigma=self.m_tool.m_update_sigma, max_iterations=self.m_tool.m_tv_training_iterations)
      ivector_trainer.initialize(m, data)

      # Performs the E-step
      ivector_trainer.e_step(ivector_machine, data)

      # write results to file
      nsamples = numpy.array([indices[1] - indices[0]], dtype=numpy.float64)

      utils.ensure_dir(os.path.dirname(stats_file))
      f = bob.io.HDF5File(stats_file, 'w')
      f.set('acc_nij_wij2', ivector_trainer.acc_nij_wij2)
      f.set('acc_fnormij_wij', ivector_trainer.acc_fnormij_wij)
      f.set('acc_nij', ivector_trainer.acc_nij)
      f.set('acc_snormij', ivector_trainer.acc_snormij)
      f.set('nsamples', nsamples)
      utils.info("IVector training: Wrote Stats file '%s'" % stats_file)


  def _read_stats(self, filename):
    """Reads accumulated IVector statistics from file"""
    utils.debug("IVector training: Reading stats file '%s'" % filename)
    f = bob.io.HDF5File(filename)
    acc_nij_wij2    = f.read('acc_nij_wij2')
    acc_fnormij_wij = f.read('acc_fnormij_wij')
    acc_nij         = f.read('acc_nij')
    acc_snormij     = f.read('acc_snormij')
    return (acc_nij_wij2, acc_fnormij_wij, acc_nij, acc_snormij)

  def ivector_mstep(self, counts, force=False):
    """Performs a single M-step of the IVector algorithm (non-parallel)"""
    old_machine_file = self.m_configuration.ivector_intermediate_file % self.m_args.iteration
    new_machine_file = self.m_configuration.ivector_intermediate_file % (self.m_args.iteration + 1)

    if  self.m_tool_chain.__check_file__(new_machine_file, force, 1000):
      utils.info("IVector training: Skipping IVector M-Step since the file '%s' already exists" % new_machine_file)
    else:
      # get the files from e-step
      training_list = self.m_file_selector.training_list('projected', 'train_projector')

      # try if there is one file containing all data
      if os.path.exists(self.m_configuration.ivector_stats_file % (self.m_args.iteration, 0, len(training_list))):
        stats_file = self.m_configuration.ivector_stats_file % (self.m_args.iteration, 0, len(training_list))
        # load stats file
        acc_nij_wij2, acc_fnormij_wij, acc_nij, acc_snormij = self._read_stats(stats_file)
      else:
        # load several files
        job_ids = range(self.__generate_job_array__(training_list, counts)[1])
        job_indices = [(counts * job_id, min(counts * (job_id+1), len(training_list))) for job_id in job_ids]
        stats_files = [self.m_configuration.ivector_stats_file % (self.m_args.iteration, indices[0], indices[1]) for indices in job_indices]

        # read all stats files
        acc_nij_wij2, acc_fnormij_wij, acc_nij, acc_snormij = self._read_stats(stats_files[0])
        for stats_file in stats_files[1:]:
          acc_nij_wij2_, acc_fnormij_wij_, acc_nij_, acc_snormij_ = self._read_stats(stats_file)
          acc_nij_wij2    += acc_nij_wij2_
          acc_fnormij_wij += acc_fnormij_wij_
          acc_nij         += acc_nij_
          acc_snormij     += acc_snormij_

      # TODO read some features (needed for computation, but not really required)
      data = []

      # Temporary machine used for initialization
      ubm = bob.machine.GMMMachine(bob.io.HDF5File(self.m_tool.m_gmm_filename))
      m = bob.machine.IVectorMachine(ubm, self.m_tool.m_subspace_dimension_of_t)
      m.variance_threshold = self.m_tool.m_variance_threshold
      # Load machine
      ivector_machine = bob.machine.IVectorMachine(bob.io.HDF5File(old_machine_file))
      ivector_machine.ubm = ubm

      # Creates the IVectorTrainer and call the initialization procedure
      ivector_trainer = bob.trainer.IVectorTrainer(update_sigma=self.m_tool.m_update_sigma, max_iterations=self.m_tool.m_tv_training_iterations)
      ivector_trainer.initialize(m, data)

      # Performs the M-step
      ivector_trainer.acc_nij_wij2 = acc_nij_wij2
      ivector_trainer.acc_fnormij_wij = acc_fnormij_wij
      ivector_trainer.acc_nij = acc_nij
      ivector_trainer.acc_snormij = acc_snormij
      ivector_trainer.m_step(ivector_machine, data) # data is not used in M-step
      utils.info("IVector training: Performed M step %d" % (self.m_args.iteration,))

      # Save the IVector model
      utils.ensure_dir(os.path.dirname(new_machine_file))
      ivector_machine.save(bob.io.HDF5File(new_machine_file, 'w'))
      shutil.copy(new_machine_file, self.m_tool.m_ivec_filename)
      utils.info("IVector training: Wrote new IVector machine '%s'" % new_machine_file)

    if self.m_args.clean_intermediate and self.m_args.iteration > 0:
      old_file = self.m_configuration.ivector_intermediate_file % (self.m_args.iteration-1)
      utils.info("Removing old intermediate directory '%s'" % os.path.dirname(old_file))
      shutil.rmtree(os.path.dirname(old_file))


  def ivector_project(self, indices, force=False):
    """Performs IVector projection"""
    # read UBM into the IVector class
    self.m_tool._load_projector_gmm_resolved(self.m_tool.m_gmm_filename)
    self.m_tool._load_projector_ivector_resolved(self.m_tool.m_ivec_filename)
      
    projected_files = self.m_file_selector.projected_list()

    # select a subset of indices to iterate
    if indices != None:
      index_range = range(indices[0], indices[1])
      utils.info("- Projection: splitting of index range %s" % str(indices))
    else:
      index_range = range(len(projected_files))

    utils.info("- Projection: projecting %d gmm stats from directory '%s' to directory '%s'" % (len(index_range), self.m_tool._resolve_projected_gmm(self.m_file_selector.projected_directory), self.m_tool._resolve_projected_ivector(self.m_file_selector.projected_directory)))
    # extract the features
    for i in index_range:
      projected_file = projected_files[i]
      projected_file_gmm_resolved = self.m_tool._resolve_projected_gmm(projected_file)
      projected_file_ivec_resolved = self.m_tool._resolve_projected_ivector(projected_file)

      if not self.m_tool_chain.__check_file__(projected_file_ivec_resolved, force):
        # load feature
        feature = self.m_tool.read_feature(str(projected_file))
        # project feature
        projected = self.m_tool._project_ivector(feature)
        # write it
        utils.ensure_dir(os.path.dirname(projected_file_ivec_resolved))
        self.m_tool._save_feature_ivector(projected, str(projected_file))

#######################################################################################
##############  Functions dealing with submission and execution of jobs  ##############
#######################################################################################

  def add_jobs_to_grid(self, external_dependencies):
    """Adds all (desired) jobs of the tool chain to the grid."""
    # collect the job ids
    job_ids = {}

    # if there are any external dependencies, we need to respect them
    deps = external_dependencies[:]

    # image preprocessing; never has any dependencies.
    if not self.m_args.skip_preprocessing:
      job_ids['preprocessing'] = self.submit_grid_job(
              'preprocess',
              list_to_split = self.m_file_selector.original_image_list(),
              number_of_files_per_job = self.m_grid_config.number_of_images_per_job,
              dependencies = [],
              **self.m_grid_config.preprocessing_queue)
      deps.append(job_ids['preprocessing'])

    # feature extraction training
    if not self.m_args.skip_extractor_training and self.m_extractor.requires_training:
      job_ids['extractor-training'] = self.submit_grid_job(
              'train-extractor',
              name = 'train-f',
              dependencies = deps,
              **self.m_grid_config.training_queue)
      deps.append(job_ids['extractor-training'])

    # feature extraction
    if not self.m_args.skip_extraction:
      job_ids['extraction'] = self.submit_grid_job(
              'extract',
              list_to_split = self.m_file_selector.preprocessed_image_list(),
              number_of_files_per_job = self.m_grid_config.number_of_features_per_job,
              dependencies = deps,
              **self.m_grid_config.extraction_queue)
      deps.append(job_ids['extraction'])

    # feature normalization
    if self.m_args.normalize_features and not self.m_args.skip_normalization:
      job_ids['feature_normalization'] = self.submit_grid_job(
              'normalize-features',
              name="norm-f",
              list_to_split = self.training_list(),
              number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
              dependencies = deps,
              **self.m_grid_config.projection_queue)
      deps.append(job_ids['feature_normalization'])

    # KMeans
    if not self.m_args.skip_k_means:
      # initialization
      if not self.m_args.kmeans_start_iteration:
        job_ids['kmeans-init'] = self.submit_grid_job(
                'kmeans-init',
                name = 'k-init',
                dependencies = deps,
                **self.m_grid_config.training_queue)
        deps.append(job_ids['kmeans-init'])

      # several iterations of E and M steps
      for iteration in range(self.m_args.kmeans_start_iteration, self.m_args.kmeans_training_iterations):
        # E-step
        job_ids['kmeans-e-step'] = self.submit_grid_job(
                'kmeans-e-step --iteration %d' % iteration,
                name='k-e-%d' % iteration,
                list_to_split = self.training_list(),
                number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
                dependencies = [job_ids['kmeans-m-step']] if iteration != self.m_args.kmeans_start_iteration else deps,
                **self.m_grid_config.projection_queue)

        # M-step
        job_ids['kmeans-m-step'] = self.submit_grid_job(
                'kmeans-m-step --iteration %d' % iteration,
                name='k-m-%d' % iteration,
                dependencies = [job_ids['kmeans-e-step']],
                **self.m_grid_config.training_queue)

      # add dependence to the last m step
      deps.append(job_ids['kmeans-m-step'])


    # GMM
    if not self.m_args.skip_gmm:
      # initialization
      if not self.m_args.gmm_start_iteration:
        job_ids['gmm-init'] = self.submit_grid_job(
                'gmm-init',
                name = 'g-init',
                dependencies = deps,
                **self.m_grid_config.training_queue)
        deps.append(job_ids['gmm-init'])

      # several iterations of E and M steps
      for iteration in range(self.m_args.gmm_start_iteration, self.m_args.gmm_training_iterations):
        # E-step
        job_ids['gmm-e-step'] = self.submit_grid_job(
                'gmm-e-step --iteration %d' % iteration,
                name='g-e-%d' % iteration,
                list_to_split = self.training_list(),
                number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
                dependencies = [job_ids['gmm-m-step']] if iteration != self.m_args.gmm_start_iteration else deps,
                **self.m_grid_config.projection_queue)

        # M-step
        job_ids['gmm-m-step'] = self.submit_grid_job(
                'gmm-m-step --iteration %d' % iteration,
                name='g-m-%d' % iteration,
                dependencies = [job_ids['gmm-e-step']],
                **self.m_grid_config.training_queue)

      # add dependence to the last m step
      deps.append(job_ids['gmm-m-step'])

    # gmm projection
    if not self.m_args.skip_gmm_projection:
      job_ids['gmm-project'] = self.submit_grid_job(
              'gmm-project',
              name = 'g-project',
              list_to_split = self.m_file_selector.feature_list(),
              number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
              dependencies = deps,
              **self.m_grid_config.projection_queue)
      deps.append(job_ids['gmm-project'])


    # I-vector
    if not self.m_args.skip_ivec:
      # initialization
      if not self.m_args.ivector_start_iteration:
        job_ids['ivec-init'] = self.submit_grid_job(
                'ivec-init',
                name = 'ivec-init',
                dependencies = deps,
                **self.m_grid_config.training_queue)
        deps.append(job_ids['ivec-init'])

      # several iterations of E and M steps
      for iteration in range(self.m_args.ivector_start_iteration, self.m_args.ivector_training_iterations):
        # E-step
        job_ids['ivec-e-step'] = self.submit_grid_job(
                'ivec-e-step --iteration %d' % iteration,
                name='ivec-e-%d' % iteration,
                list_to_split = self.training_list(),
                number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
                dependencies = [job_ids['ivec-m-step']] if iteration != self.m_args.ivector_start_iteration else deps,
                **self.m_grid_config.projection_queue)

        # M-step
        job_ids['ivec-m-step'] = self.submit_grid_job(
                'ivec-m-step --iteration %d' % iteration,
                name='ivec-m-%d' % iteration,
                dependencies = [job_ids['ivec-e-step']],
                **self.m_grid_config.training_queue)

      # add dependence to the last m step
      deps.append(job_ids['ivec-m-step'])
   
 
    # ivec projection
    if not self.m_args.skip_ivec_projection:
      job_ids['ivec-project'] = self.submit_grid_job(
              'ivec-project',
              name = 'ivec-project',
              list_to_split = self.m_file_selector.projected_list(),
              number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
              dependencies = deps,
              **self.m_grid_config.projection_queue)
      deps.append(job_ids['ivec-project'])

    # return the job ids, in case anyone wants to know them
    return job_ids


  def execute_grid_job(self):
    """Run the desired job of the ZT tool chain that is specified on command line."""
    # preprocess the images
    if self.m_args.sub_task == 'preprocess':
      self.m_tool_chain.preprocess_images(
          self.m_preprocessor,
          indices = self.indices(self.m_file_selector.original_image_list(), self.m_grid_config.number_of_images_per_job),
          force = self.m_args.force)

    # train the feature extractor
    elif self.m_args.sub_task == 'train-extractor':
      self.m_tool_chain.train_extractor(
          self.m_extractor,
          self.m_preprocessor,
          force = self.m_args.force)

    # extract the features
    elif self.m_args.sub_task == 'extract':
      self.m_tool_chain.extract_features(
          self.m_extractor,
          self.m_preprocessor,
          indices = self.indices(self.m_file_selector.preprocessed_image_list(), self.m_grid_config.number_of_features_per_job),
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'normalize-features':
      self.feature_normalization(
          indices = self.indices(self.training_list(), self.m_grid_config.number_of_projections_per_job),
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'kmeans-init':
      self.kmeans_initialize(
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'kmeans-e-step':
      self.kmeans_estep(
          indices = self.indices(self.training_list(), self.m_grid_config.number_of_projections_per_job),
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'kmeans-m-step':
      self.kmeans_mstep(
          counts = self.m_grid_config.number_of_projections_per_job,
          force = self.m_args.force)

    elif self.m_args.sub_task == 'gmm-init':
      self.gmm_initialize(
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'gmm-e-step':
      self.gmm_estep(
          indices = self.indices(self.training_list(), self.m_grid_config.number_of_projections_per_job),
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'gmm-m-step':
      self.gmm_mstep(
          counts = self.m_grid_config.number_of_projections_per_job,
          force = self.m_args.force)

    # project using the gmm ubm
    elif self.m_args.sub_task == 'gmm-project':
      self.gmm_project(
          indices = self.indices(self.m_file_selector.feature_list(), self.m_grid_config.number_of_projections_per_job),
          force = self.m_args.force)

    # I-vector initialization
    elif self.m_args.sub_task == 'ivec-init':
      self.ivector_initialize(
          force = self.m_args.force)

    # I-vector e-step
    elif self.m_args.sub_task == 'ivec-e-step':
      self.ivector_estep(
          indices = self.indices(self.training_list(), self.m_grid_config.number_of_projections_per_job),
          force = self.m_args.force)

    # I-vector m-step
    elif self.m_args.sub_task == 'ivec-m-step':
      self.ivector_mstep(
          counts = self.m_grid_config.number_of_projections_per_job,
          force = self.m_args.force)

    # project using ivec
    elif self.m_args.sub_task == 'ivec-project':
      self.ivector_project(
          indices = self.indices(self.m_file_selector.projected_list(), self.m_grid_config.number_of_projections_per_job),
          force = self.m_args.force)

    # Test if the keyword was processed
    else:
      raise ValueError("The given subtask '%s' could not be processed. THIS IS A BUG. Please report this to the authors." % self.m_args.sub_task)


def parse_args(command_line_parameters):
  """This function parses the given options (which by default are the command line options)."""
  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      conflict_handler='resolve')

  # add the arguments required for all tool chains
  config_group, dir_group, file_group, sub_dir_group, other_group, skip_group = ToolChainExecutorIVector.required_command_line_options(parser)

  config_group.add_argument('-P', '--protocol', metavar='PROTOCOL',
      help = 'Overwrite the protocol that is stored in the database by the given one (might not by applicable for all databases).')
  config_group.add_argument('-t', '--tool', metavar = 'x', nargs = '+', default = ['ivector'],
      help = 'IVector-based face recognition; registered face recognition tools are: %s'%utils.resources.resource_keys('tool'))
  config_group.add_argument('-g', '--grid', metavar = 'x', required=True,
      help = 'Configuration file for the grid setup; needs to be specified.')

  #######################################################################################
  ############################ other options ############################################
  other_group.add_argument('-F', '--force', action='store_true',
      help = 'Force to erase former data if already exist')

  other_group.add_argument('-l', '--limit-training-examples', type=int,
      help = 'Limit the number of training examples used for KMeans initialization and the GMM initialization')

  other_group.add_argument('-K', '--kmeans-training-iterations', type=int, default=10,
      help = 'Specify the number of training iterations for the KMeans training')
  other_group.add_argument('-k', '--kmeans-start-iteration', type=int, default=0,
      help = 'Specify the first iteration for the KMeans training (i.e. to restart)')

  other_group.add_argument('-M', '--gmm-training-iterations', type=int, default=10,
      help = 'Specify the number of training iterations for the GMM training')
  other_group.add_argument('-m', '--gmm-start-iteration', type=int, default=0,
      help = 'Specify the first iteration for the GMM training (i.e. to restart)')
  other_group.add_argument('-n', '--normalize-features', action='store_true',
      help = 'Normalize features before IVector training?')
  other_group.add_argument('-C', '--clean-intermediate', action='store_true',
      help = 'Clean up temporary files of older iterations?')

  other_group.add_argument('-Q', '--ivector-training-iterations', type=int, default=25,
      help = 'Specify the number of training iterations for the IVector training')
  other_group.add_argument('-q', '--ivector-start-iteration', type=int, default=0,
      help = 'Specify the first iteration for the IVector training (i.e. to restart)')

  skip_group.add_argument('--skip-normalization', '--non', action='store_true',
      help = "Skip the feature normalization step")
  skip_group.add_argument('--skip-k-means', '--nok', action='store_true',
      help = "Skip the KMeans step")
  skip_group.add_argument('--skip-gmm', '--nog', action='store_true',
      help = "Skip the GMM step")
  skip_group.add_argument('--skip-gmm-projection', '--nogp', action='store_true',
      help = "Skip the GMM projection step")
  skip_group.add_argument('--skip-ivec', '--noi', action='store_true',
      help = "Skip the IVector step")
  skip_group.add_argument('--skip-ivec-projection', '--noip', action='store_true',
      help = "Skip the GMM ivec projection")

  #######################################################################################
  #################### sub-tasks being executed by this script ##########################
  parser.add_argument('--sub-task',
      choices = ('preprocess', 'train-extractor', 'extract', 'normalize-features', 'kmeans-init', 'kmeans-e-step', 'kmeans-m-step', 'gmm-init', 'gmm-e-step', 'gmm-m-step', 'gmm-project',  'ivec-init', 'ivec-e-step', 'ivec-m-step', 'ivec-project'),
      help = argparse.SUPPRESS) #'Executes a subtask (FOR INTERNAL USE ONLY!!!)'
  parser.add_argument('--iteration', type=int,
      help = argparse.SUPPRESS) #'The current iteration of KMeans or GMM training'

  return parser.parse_args(command_line_parameters)


def face_verify(args, command_line_parameters, external_dependencies = [], external_fake_job_id = 0):
  """This is the main entry point for computing face verification experiments.
  You just have to specify configuration scripts for any of the steps of the toolchain, which are:
  -- the database
  -- feature extraction (including image preprocessing)
  -- the score computation tool
  -- and the grid configuration (in case, the function should be executed in the grid).
  Additionally, you can skip parts of the toolchain by selecting proper --skip-... parameters.
  If your probe files are not too big, you can also specify the --preload-probes switch to speed up the score computation.
  If files should be re-generated, please specify the --force option (might be combined with the --skip-... options)."""


  # generate tool chain executor
  executor = ToolChainExecutorIVector(args)
  # as the main entry point, check whether the grid option was given
  if args.sub_task:
    # execute the desired sub-task
    try:
      executor.execute_grid_job()
    except:
      exc_info = sys.exc_info()
      # delete dependent grid jobs, if desired
      executor.delete_dependent_grid_jobs()
      raise exc_info[1], None, exc_info[2]
    return {}
  else:
    # no other parameter given, so deploy new jobs

    # get the name of this file
    this_file = __file__
    if this_file[-1] == 'c':
      this_file = this_file[0:-1]

    # initialize the executor to submit the jobs to the grid
    executor.set_common_parameters(calling_file = this_file, parameters = command_line_parameters, fake_job_id = external_fake_job_id)

    # add the jobs
    return executor.add_jobs_to_grid(external_dependencies)


def main(command_line_parameters = sys.argv):
  """Executes the main function"""
  # do the command line parsing
  args = parse_args(command_line_parameters[1:])

  # perform face verification test
  face_verify(args, command_line_parameters)

if __name__ == "__main__":
  main()


