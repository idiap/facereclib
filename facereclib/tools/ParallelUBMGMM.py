#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury <Elie.Khoury@idiap.ch>
# Manuel Guenther <Manuel.Guenther@idiap.ch>
# Tue Aug 27 22:40:41 CEST 2013
#
# Copyright (C) 2012-2013 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import sys, os, shutil
import argparse
import bob
import numpy
from . import UBMGMM
from .. import utils

class ParallelUBMGMM():
  
  def __init__(self):
    pass
    
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

    utils.info("UBM training: normalizing features from range(%d, %d)" % indices)

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
      utils.info("UBM training: Skipping KMeans initialization since the file '%s' already exists" % output_file)
    else:
      # read data
      utils.info("UBM training: initializing kmeans")
      training_list = self.training_list()
      data = numpy.vstack([self.m_extractor.read_feature(str(training_list[index])) for index in utils.quasi_random_indices(len(training_list), self.m_args.limit_training_examples)])

      # Perform KMeans initialization
      kmeans_machine = bob.machine.KMeansMachine(self.m_tool.m_gaussians, data.shape[1])
      # Creates the KMeansTrainer and call the initialization procedure
      kmeans_trainer = bob.trainer.KMeansTrainer()
      kmeans_trainer.initialize(kmeans_machine, data)
      utils.ensure_dir(os.path.dirname(output_file))
      kmeans_machine.save(bob.io.HDF5File(output_file, 'w'))
      utils.info("UBM training: saved initial KMeans machine to '%s'" % output_file)



  def kmeans_estep(self, indices, force=False):
    """Performs a single E-step of the K-Means algorithm (parallel)"""
    stats_file = self.m_configuration.kmeans_stats_file % (self.m_args.iteration, indices[0], indices[1])

    if  self.m_tool_chain.__check_file__(stats_file, force, 1000):
      utils.info("UBM training: Skipping KMeans E-Step since the file '%s' already exists" % stats_file)
    else:
      training_list = self.training_list()
      machine_file = self.m_configuration.kmeans_intermediate_file % self.m_args.iteration
      kmeans_machine = bob.machine.KMeansMachine(bob.io.HDF5File(machine_file))

      utils.info("UBM training: KMeans E-Step from range(%d, %d)" % indices)

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
      utils.info("UBM training: Wrote Stats file '%s'" % stats_file)



  def read_stats(self, filename):
    """Reads accumulated K-Means statistics from file"""
    utils.debug("UBM training: Reading stats file '%s'" % filename)
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
      utils.info("UBM training: Skipping KMeans M-Step since the file '%s' already exists" % new_machine_file)
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
      utils.info("UBM training: Performed M step %d with result %f" % (self.m_args.iteration, dist/nsamples))

      # Save the K-Means model
      utils.ensure_dir(os.path.dirname(new_machine_file))
      kmeans_machine.save(bob.io.HDF5File(new_machine_file, 'w'))
      shutil.copy(new_machine_file, self.m_configuration.kmeans_file)
      utils.info("UBM training: Wrote new KMeans machine '%s'" % new_machine_file)

    if self.m_args.clean_intermediate and self.m_args.iteration > 0:
      old_file = self.m_configuration.kmeans_intermediate_file % (self.m_args.iteration-1)
      utils.info("Removing old intermediate directory '%s'" % os.path.dirname(old_file))
      shutil.rmtree(os.path.dirname(old_file))



  def gmm_initialize(self, force=False):
    """Initializes the GMM calculation with the result of the K-Means algorithm (non-parallel).
    This might require a lot of memory."""
    output_file = self.m_configuration.gmm_intermediate_file % 0

    if self.m_tool_chain.__check_file__(output_file, force, 800):
      utils.info("UBM Training: Skipping GMM initialization since '%s' already exists" % output_file)
    else:
      training_list = self.training_list()
      utils.info("UBM Training: Initializing GMM")

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
      utils.info("UBM Training: Wrote GMM file '%s'" % output_file)


  def gmm_estep(self, indices, force=False):
    """Performs a single E-step of the GMM training (parallel)."""
    stats_file = self.m_configuration.gmm_stats_file % (self.m_args.iteration, indices[0], indices[1])

    if  self.m_tool_chain.__check_file__(stats_file, force, 1000):
      utils.info("UBM training: Skipping GMM E-Step since the file '%s' already exists" % stats_file)
    else:
      training_list = self.training_list()
      machine_file = self.m_configuration.gmm_intermediate_file % self.m_args.iteration
      gmm_machine = bob.machine.GMMMachine(bob.io.HDF5File(machine_file))

      utils.info("UBM training: GMM E-Step from range(%d, %d)" % indices)

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
      utils.info("UBM training: Wrote GMM stats '%s'" % (stats_file))


  def gmm_mstep(self, counts, force=False):
    """Performs a single M-step of the GMM training (non-parallel)"""
    old_machine_file = self.m_configuration.gmm_intermediate_file % self.m_args.iteration
    new_machine_file = self.m_configuration.gmm_intermediate_file % (self.m_args.iteration + 1)

    if  self.m_tool_chain.__check_file__(new_machine_file, force, 1000):
      utils.info("UBM training: Skipping GMM M-Step since the file '%s' already exists" % new_machine_file)
    else:
      # get the files from e-step
      training_list = self.training_list()

      # try if there is one file containing all data
      if os.path.exists(self.m_configuration.gmm_stats_file % (self.m_args.iteration, 0, len(training_list))):
        stats_file = self.m_configuration.gmm_stats_file % (self.m_args.iteration, 0, len(training_list))
        # load stats file
        gmm_stats = bob.machine.GMMStats(bob.io.HDF5File(stats_file))
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
    utils.info("- Projection: projecting %d features from directory '%s' to directory '%s'" % (len(index_range), self.m_file_selector.features_directory, self.m_tool._resolve_projected_gmm(self.m_file_selector.projected_directory)))
    # extract the features
    for i in index_range:
      feature_file = feature_files[i]
      projected_file = projected_files[i]
      projected_file_resolved = self.m_tool._resolve_projected_gmm(projected_file)

      if not self.m_tool_chain.__check_file__(projected_file_resolved, force):
        # load feature
        feature = self.m_extractor.read_feature(str(feature_file))
        # project feature
        projected = UBMGMM.project(self.m_tool, feature)
        # write it
        utils.ensure_dir(os.path.dirname(projected_file_resolved))
        self.m_tool._save_feature_gmm(projected, str(projected_file))

