#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
# Elie Khoury <Elie.Khoury@idiap.ch>
# Wed Aug 28 14:51:13 CEST 2013
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
#

import sys, os, shutil
import argparse
import bob
import numpy

from . import ToolChainExecutor
from .. import toolchain, tools, utils


class ToolChainExecutorISV (ToolChainExecutor.ToolChainExecutor, tools.ParallelUBMGMM):
  """Class that executes the ZT tool chain (locally or in the grid)."""

  def __init__(self, args):
    # call base class constructor
    ToolChainExecutor.ToolChainExecutor.__init__(self, args)
    tools.ParallelUBMGMM.__init__(self)
    
    if not isinstance(self.m_tool, tools.ISV):
      raise ValueError("This script is specifically designed to compute ISV tests. Please select an according tool.")

    self.m_tool.m_gmm_isv_split = True

    if args.protocol:
      self.m_database.protocol = args.protocol

    self.m_configuration.normalized_directory = os.path.join(self.m_configuration.temp_directory, 'normalized_features')
    self.m_configuration.kmeans_file = os.path.join(self.m_configuration.temp_directory, 'k_means.hdf5')
    self.m_configuration.kmeans_intermediate_file = os.path.join(self.m_configuration.temp_directory, 'kmeans_temp', 'i_%05d', 'k_means.hdf5')
    self.m_configuration.kmeans_stats_file = os.path.join(self.m_configuration.temp_directory, 'kmeans_temp', 'i_%05d', 'stats_%05d-%05d.hdf5')
    self.m_tool.m_gmm_filename = os.path.join(self.m_configuration.temp_directory, 'gmm.hdf5')
    self.m_configuration.gmm_intermediate_file = os.path.join(self.m_configuration.temp_directory, 'gmm_temp', 'i_%05d', 'gmm.hdf5')
    self.m_configuration.gmm_stats_file = os.path.join(self.m_configuration.temp_directory, 'gmm_temp', 'i_%05d', 'stats_%05d-%05d.hdf5')
    self.m_tool.m_isv_filename = os.path.join(self.m_configuration.temp_directory, 'isv.hdf5')
    self.m_tool.m_projected_toreplace = 'projected'
    self.m_tool.m_projected_gmm = 'projected_gmm'
    self.m_tool.m_projected_isv = 'projected_isv'
    self.m_tool.m_projector_toreplace = self.m_configuration.projector_file


    self.m_configuration.models_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.models_directories[0], self.m_database.protocol)
    self.m_configuration.scores_no_norm_directory = os.path.join(self.m_configuration.user_directory, self.m_args.score_sub_directory, self.m_database.protocol, self.m_args.zt_score_directories[0])
    # add specific configuration for ZT-normalization
    if args.zt_norm:
      self.m_configuration.t_norm_models_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.models_directories[1], self.m_database.protocol)
      models_directories = (self.m_configuration.models_directory, self.m_configuration.t_norm_models_directory)

      self.m_configuration.scores_zt_norm_directory = os.path.join(self.m_configuration.user_directory, self.m_args.score_sub_directory, self.m_database.protocol, self.m_args.zt_score_directories[1])
      score_directories = (self.m_configuration.scores_no_norm_directory, self.m_configuration.scores_zt_norm_directory)

      self.m_configuration.zt_norm_A_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.score_sub_directory, self.m_database.protocol, self.m_args.zt_temp_directories[0])
      self.m_configuration.zt_norm_B_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.score_sub_directory, self.m_database.protocol, self.m_args.zt_temp_directories[1])
      self.m_configuration.zt_norm_C_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.score_sub_directory, self.m_database.protocol, self.m_args.zt_temp_directories[2])
      self.m_configuration.zt_norm_D_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.score_sub_directory, self.m_database.protocol, self.m_args.zt_temp_directories[3])
      self.m_configuration.zt_norm_D_sameValue_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.score_sub_directory, self.m_database.protocol, self.m_args.zt_temp_directories[4])
      zt_score_directories = (self.m_configuration.zt_norm_A_directory, self.m_configuration.zt_norm_B_directory, self.m_configuration.zt_norm_C_directory, self.m_configuration.zt_norm_D_directory, self.m_configuration.zt_norm_D_sameValue_directory)
    else:
      models_directories = (self.m_configuration.models_directory,)
      score_directories = (self.m_configuration.scores_no_norm_directory,)
      zt_score_directories = None

    # specify the file selector to be used
    self.m_file_selector = toolchain.FileSelector(
        self.m_database,
        preprocessed_directory = self.m_configuration.preprocessed_directory,
        extractor_file = self.m_configuration.extractor_file,
        features_directory = self.m_configuration.features_directory,
        projector_file = self.m_configuration.projector_file,
        projected_directory = self.m_configuration.projected_directory,
        enroller_file = self.m_configuration.enroller_file,
        model_directories = models_directories,
        score_directories = score_directories,
        zt_score_directories = zt_score_directories
    )

    # create the tool chain to be used to actually perform the parts of the experiments
    self.m_tool_chain = toolchain.ToolChain(self.m_file_selector)




#######################################################################################
####################  Functions that will be executed in the grid  ####################
#######################################################################################

  def isv_training(self, force=False):
    """Finally, the UBM is used to train the ISV projector/enroller."""
    if self.m_tool_chain.__check_file__(self.m_tool.m_isv_filename, force, 800):
      utils.info("ISV training: Skipping ISV training since '%s' already exists" % self.m_tool.m_isv_filename)
    else:
      # read UBM into the ISV class
      self.m_tool.m_ubm = bob.machine.GMMMachine(bob.io.HDF5File(self.m_tool.m_gmm_filename))

      # read training data
      training_list = self.m_file_selector.training_list('projected', 'train_projector', arrange_by_client = True)
      train_features = self.m_tool_chain.__read_features_by_client__(training_list, self.m_tool)

      # perform ISV training
      utils.info("ISV training: training ISV with %d clients" % len(train_features))
      self.m_tool._train_isv(train_features)
      utils.ensure_dir(os.path.dirname(self.m_tool.m_isv_filename))
      self.m_tool._save_projector_isv_resolved(self.m_tool.m_isv_filename)
      utils.info("ISV training: saved ISV matrix to '%s'" % self.m_tool.m_isv_filename)
      self.m_tool._save_projector_together(self.m_configuration.projector_file)


  def isv_project(self, indices, force=False):
    """Performs ISV projection"""
    # read UBM into the ISV class
    self.m_tool._load_projector_gmm_resolved(self.m_tool.m_gmm_filename)
    #isv_file = self.m_configuration.projector_file
    self.m_tool._load_projector_isv_resolved(self.m_tool.m_isv_filename)

    projected_files = self.m_file_selector.projected_list()

    # select a subset of indices to iterate
    if indices != None:
      index_range = range(indices[0], indices[1])
      utils.info("- Projection: splitting of index range %s" % str(indices))
    else:
      index_range = range(len(projected_files))

    utils.info("- Projection: projecting %d gmm stats from directory '%s' to directory '%s'" % (len(index_range), self.m_tool._resolve_projected_gmm(self.m_file_selector.projected_directory), self.m_tool._resolve_projected_isv(self.m_file_selector.projected_directory)))
    # extract the features
    for i in index_range:
      projected_file = projected_files[i]
      projected_file_gmm_resolved = self.m_tool._resolve_projected_gmm(projected_file)
      projected_file_isv_resolved = self.m_tool._resolve_projected_isv(projected_file)

      if not self.m_tool_chain.__check_file__(projected_file_isv_resolved, force):
        # load feature
        feature = self.m_tool.read_feature(str(projected_file))
        # project feature
        projected = self.m_tool._project_isv(feature)
        # write it
        utils.ensure_dir(os.path.dirname(projected_file_isv_resolved))
        self.m_tool._save_feature_isv(projected, str(projected_file))

#######################################################################################
##############  Functions dealing with submission and execution of jobs  ##############
#######################################################################################

  def add_jobs_to_grid(self, external_dependencies):
    """Adds all (desired) jobs of the tool chain to the grid."""
    # collect the job ids
    job_ids = {}

    # if there are any external dependencies, we need to respect them
    deps = external_dependencies[:]

    # preprocessing; never has any dependencies.
    if not self.m_args.skip_preprocessing:
      job_ids['preprocessing'] = self.submit_grid_job(
              'preprocess',
              list_to_split = self.m_file_selector.original_data_list(),
              number_of_files_per_job = self.m_grid.number_of_preprocessings_per_job,
              dependencies = [],
              **self.m_grid.preprocessing_queue)
      deps.append(job_ids['preprocessing'])

    # feature extraction training
    if not self.m_args.skip_extractor_training and self.m_extractor.requires_training:
      job_ids['extractor-training'] = self.submit_grid_job(
              'train-extractor',
              name = 'train-f',
              dependencies = deps,
              **self.m_grid.training_queue)
      deps.append(job_ids['extractor-training'])

    # feature extraction
    if not self.m_args.skip_extraction:
      job_ids['extraction'] = self.submit_grid_job(
              'extract',
              list_to_split = self.m_file_selector.preprocessed_data_list(),
              number_of_files_per_job = self.m_grid.number_of_extracted_features_per_job,
              dependencies = deps,
              **self.m_grid.extraction_queue)
      deps.append(job_ids['extraction'])

    # feature normalization
    if self.m_args.normalize_features and not self.m_args.skip_normalization:
      job_ids['feature_normalization'] = self.submit_grid_job(
              'normalize-features',
              name="norm-f",
              list_to_split = self.training_list(),
              number_of_files_per_job = self.m_grid.number_of_projected_features_per_job,
              dependencies = deps,
              **self.m_grid.projection_queue)
      deps.append(job_ids['feature_normalization'])

    # KMeans
    if not self.m_args.skip_k_means:
      # initialization
      if not self.m_args.kmeans_start_iteration:
        job_ids['kmeans-init'] = self.submit_grid_job(
                'kmeans-init',
                name = 'k-init',
                dependencies = deps,
                **self.m_grid.training_queue)
        deps.append(job_ids['kmeans-init'])

      # several iterations of E and M steps
      for iteration in range(self.m_args.kmeans_start_iteration, self.m_args.kmeans_training_iterations):
        # E-step
        job_ids['kmeans-e-step'] = self.submit_grid_job(
                'kmeans-e-step --iteration %d' % iteration,
                name='k-e-%d' % iteration,
                list_to_split = self.training_list(),
                number_of_files_per_job = self.m_grid.number_of_projected_features_per_job,
                dependencies = [job_ids['kmeans-m-step']] if iteration != self.m_args.kmeans_start_iteration else deps,
                **self.m_grid.projection_queue)

        # M-step
        job_ids['kmeans-m-step'] = self.submit_grid_job(
                'kmeans-m-step --iteration %d' % iteration,
                name='k-m-%d' % iteration,
                dependencies = [job_ids['kmeans-e-step']],
                **self.m_grid.training_queue)

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
                **self.m_grid.training_queue)
        deps.append(job_ids['gmm-init'])

      # several iterations of E and M steps
      for iteration in range(self.m_args.gmm_start_iteration, self.m_args.gmm_training_iterations):
        # E-step
        job_ids['gmm-e-step'] = self.submit_grid_job(
                'gmm-e-step --iteration %d' % iteration,
                name='g-e-%d' % iteration,
                list_to_split = self.training_list(),
                number_of_files_per_job = self.m_grid.number_of_projected_features_per_job,
                dependencies = [job_ids['gmm-m-step']] if iteration != self.m_args.gmm_start_iteration else deps,
                **self.m_grid.projection_queue)

        # M-step
        job_ids['gmm-m-step'] = self.submit_grid_job(
                'gmm-m-step --iteration %d' % iteration,
                name='g-m-%d' % iteration,
                dependencies = [job_ids['gmm-e-step']],
                **self.m_grid.training_queue)

      # add dependence to the last m step
      deps.append(job_ids['gmm-m-step'])

    # gmm projection
    if not self.m_args.skip_gmm_projection:
      job_ids['gmm-project'] = self.submit_grid_job(
              'gmm-project',
              name = 'g-project',
              list_to_split = self.m_file_selector.feature_list(),
              number_of_files_per_job = self.m_grid.number_of_projected_features_per_job,
              dependencies = deps,
              **self.m_grid.projection_queue)
      deps.append(job_ids['gmm-project'])

    # feature projection training
    if not self.m_args.skip_isv:
      # check if we have a special queue for the ISV training (which usually needs a lot of memory)
      queue = self.m_grid.isv_training_queue if hasattr(self.m_grid, 'isv_training_queue') else self.m_grid.training_queue
      job_ids['isv_training'] = self.submit_grid_job(
              'train-isv',
              name="isv",
              dependencies = deps,
              **queue)
      deps.append(job_ids['isv_training'])

    # isv projection
    if not self.m_args.skip_isv_projection:
      job_ids['isv-project'] = self.submit_grid_job(
              'isv-project',
              name = 'isv-project',
              list_to_split = self.m_file_selector.projected_list(),
              number_of_files_per_job = self.m_grid.number_of_projected_features_per_job,
              dependencies = deps,
              **self.m_grid.projection_queue)
      deps.append(job_ids['isv-project'])

    # enroll models
    enroll_deps_n = {}
    enroll_deps_t = {}
    score_deps = {}
    concat_deps = {}
    for group in self.m_args.groups:
      enroll_deps_n[group] = deps[:]
      enroll_deps_t[group] = deps[:]
      if not self.m_args.skip_enrollment:
        job_ids['enroll_%s_N'%group] = self.submit_grid_job(
                'enroll --group %s --model-type N'%group,
                name = "enr-N-%s"%group,
                list_to_split = self.m_file_selector.model_ids(group),
                number_of_files_per_job = self.m_grid.number_of_enrolled_models_per_job,
                dependencies = deps,
                **self.m_grid.enrollment_queue)
        enroll_deps_n[group].append(job_ids['enroll_%s_N'%group])

        if self.m_args.zt_norm:
          job_ids['enroll_%s_T'%group] = self.submit_grid_job(
                  'enroll --group %s --model-type T'%group,
                  name = "enr-T-%s"%group,
                  list_to_split = self.m_file_selector.t_model_ids(group),
                  number_of_files_per_job = self.m_grid.number_of_models_per_enroll_job,
                  dependencies = deps,
                  **self.m_grid.enrollment_queue)
          enroll_deps_t[group].append(job_ids['enroll_%s_T'%group])

      # compute A,B,C, and D scores
      if not self.m_args.skip_score_computation:
        job_ids['score_%s_A'%group] = self.submit_grid_job(
                'compute-scores --group %s --score-type A'%group,
                name = "score-A-%s"%group,
                list_to_split = self.m_file_selector.model_ids(group),
                number_of_files_per_job = self.m_grid.number_of_models_per_scoring_job,
                dependencies = enroll_deps_n[group],
                **self.m_grid.scoring_queue)
        concat_deps[group] = [job_ids['score_%s_A'%group]]

        if self.m_args.zt_norm:
          job_ids['score_%s_B'%group] = self.submit_grid_job(
                  'compute-scores --group %s --score-type B'%group,
                  name = "score-B-%s"%group,
                  list_to_split = self.m_file_selector.model_ids(group),
                  number_of_files_per_job = self.m_grid.number_of_models_per_scoring_job,
                  dependencies = enroll_deps_n[group],
                  **self.m_grid.scoring_queue)

          job_ids['score_%s_C'%group] = self.submit_grid_job(
                  'compute-scores --group %s --score-type C'%group,
                  name = "score-C-%s"%group,
                  list_to_split = self.m_file_selector.t_model_ids(group),
                  number_of_files_per_job = self.m_grid.number_of_models_per_scoring_job,
                  dependencies = enroll_deps_t[group],
                  **self.m_grid.scoring_queue)

          job_ids['score_%s_D'%group] = self.submit_grid_job(
                  'compute-scores --group %s --score-type D'%group,
                  name = "score-D-%s"%group,
                  list_to_split = self.m_file_selector.t_model_ids(group),
                  number_of_files_per_job = self.m_grid.number_of_models_per_scoring_job,
                  dependencies = enroll_deps_t[group],
                  **self.m_grid.scoring_queue)

          # compute zt-norm
          score_deps[group] = [job_ids['score_%s_A'%group], job_ids['score_%s_B'%group], job_ids['score_%s_C'%group], job_ids['score_%s_D'%group]]
          job_ids['score_%s_Z'%group] = self.submit_grid_job(
                  'compute-scores --group %s --score-type Z'%group,
                  name = "score-Z-%s"%group,
                  dependencies = score_deps[group])
          concat_deps[group].extend([job_ids['score_%s_B'%group], job_ids['score_%s_C'%group], job_ids['score_%s_D'%group], job_ids['score_%s_Z'%group]])
      else:
        concat_deps[group] = []

      # concatenate results
      if not self.m_args.skip_concatenation:
        job_ids['concat_%s'%group] = self.submit_grid_job(
                'concatenate --group %s'%group,
                name = "concat-%s"%group,
                dependencies = concat_deps[group])

    # return the job ids, in case anyone wants to know them
    return job_ids


  def execute_grid_job(self):
    """Run the desired job of the ZT tool chain that is specified on command line."""
    # preprocess the data
    if self.m_args.sub_task == 'preprocess':
      self.m_tool_chain.preprocess_data(
          self.m_preprocessor,
          indices = self.indices(self.m_file_selector.original_data_list(), self.m_grid.number_of_preprocessings_per_job),
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
          indices = self.indices(self.m_file_selector.preprocessed_data_list(), self.m_grid.number_of_extracted_features_per_job),
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'normalize-features':
      self.feature_normalization(
          indices = self.indices(self.training_list(), self.m_grid.number_of_projected_features_per_job),
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'kmeans-init':
      self.kmeans_initialize(
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'kmeans-e-step':
      self.kmeans_estep(
          indices = self.indices(self.training_list(), self.m_grid.number_of_projected_features_per_job),
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'kmeans-m-step':
      self.kmeans_mstep(
          counts = self.m_grid.number_of_projected_features_per_job,
          force = self.m_args.force)

    elif self.m_args.sub_task == 'gmm-init':
      self.gmm_initialize(
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'gmm-e-step':
      self.gmm_estep(
          indices = self.indices(self.training_list(), self.m_grid.number_of_projected_features_per_job),
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'gmm-m-step':
      self.gmm_mstep(
          counts = self.m_grid.number_of_projected_features_per_job,
          force = self.m_args.force)

    # project using the gmm ubm
    elif self.m_args.sub_task == 'gmm-project':
      self.gmm_project(
          indices = self.indices(self.m_file_selector.feature_list(), self.m_grid.number_of_projected_features_per_job),
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'train-isv':
      self.isv_training(
          force = self.m_args.force)

    # project using isv
    elif self.m_args.sub_task == 'isv-project':
      self.isv_project(
          indices = self.indices(self.m_file_selector.projected_list(), self.m_grid.number_of_projected_features_per_job),
          force = self.m_args.force)

    # enroll the models
    elif self.m_args.sub_task == 'enroll':
      if self.m_args.model_type == 'N':
        self.m_tool_chain.enroll_models(
            self.m_tool,
            self.m_extractor,
            self.m_args.zt_norm,
            indices = self.indices(self.m_file_selector.model_ids(self.m_args.group), self.m_grid.number_of_enrolled_models_per_job),
            groups = [self.m_args.group],
            types = ['N'],
            force = self.m_args.force)

      else:
        self.m_tool_chain.enroll_models(
            self.m_tool,
            self.m_extractor,
            self.m_args.zt_norm,
            indices = self.indices(self.m_file_selector.t_model_ids(self.m_args.group), self.m_grid.number_of_enrolled_models_per_job),
            groups = [self.m_args.group],
            types = ['T'],
            force = self.m_args.force)

    # compute scores
    elif self.m_args.sub_task == 'compute-scores':
      if self.m_args.score_type in ['A', 'B']:
        self.m_tool_chain.compute_scores(
            self.m_tool,
            self.m_args.zt_norm,
            indices = self.indices(self.m_file_selector.model_ids(self.m_args.group), self.m_grid.number_of_models_per_scoring_job),
            groups = [self.m_args.group],
            types = [self.m_args.score_type],
            preload_probes = self.m_args.preload_probes,
            force = self.m_args.force)

      elif self.m_args.score_type in ['C', 'D']:
        self.m_tool_chain.compute_scores(
            self.m_tool,
            self.m_args.zt_norm,
            indices = self.indices(self.m_file_selector.t_model_ids(self.m_args.group), self.m_grid.number_of_models_per_scoring_job),
            groups = [self.m_args.group],
            types = [self.m_args.score_type],
            preload_probes = self.m_args.preload_probes,
            force = self.m_args.force)

      else:
        self.m_tool_chain.zt_norm(groups = [self.m_args.group])

    # concatenate
    elif self.m_args.sub_task == 'concatenate':
      self.m_tool_chain.concatenate(
          self.m_args.zt_norm,
          groups = [self.m_args.group])

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
  config_group, dir_group, file_group, sub_dir_group, other_group, skip_group = ToolChainExecutorISV.required_command_line_options(parser)

  config_group.add_argument('-P', '--protocol', metavar='PROTOCOL',
      help = 'Overwrite the protocol that is stored in the database by the given one (might not by applicable for all databases).')
  config_group.add_argument('-t', '--tool', metavar = 'x', nargs = '+', default = ['isv'],
      help = 'ISV-based face recognition; registered face recognition tools are: %s'%utils.resources.resource_keys('tool'))
  config_group.add_argument('-g', '--grid', metavar = 'x', nargs = '+', required = True,
      help = 'Configuration file for the grid setup; needs to be specified.')

  sub_dir_group.add_argument('--models-directories', metavar = 'DIR', nargs = 2,
      default = ['models', 'tmodels'],
      help = 'Sub-directories (of --temp-directory) where the models should be stored')
  sub_dir_group.add_argument('--zt-temp-directories', metavar = 'DIR', nargs = 5,
      default = ['zt_norm_A', 'zt_norm_B', 'zt_norm_C', 'zt_norm_D', 'zt_norm_D_sameValue'],
      help = 'Sub-directories (of --temp-directory) where to write the ZT-norm values')
  sub_dir_group.add_argument('--zt-score-directories', metavar = 'DIR', nargs = 2,
      default = ['nonorm', 'ztnorm'],
      help = 'Sub-directories (of --user-directory) where to write the results to')

  #######################################################################################
  ############################ other options ############################################
  other_group.add_argument('-z', '--zt-norm', action='store_true',
      help = 'Enable the computation of ZT norms')
  other_group.add_argument('-F', '--force', action='store_true',
      help = 'Force to erase former data if already exist')
  other_group.add_argument('-w', '--preload-probes', action='store_true',
      help = 'Preload probe files during score computation (needs more memory, but is faster and requires fewer file accesses). WARNING! Use this flag with care!')
  other_group.add_argument('--groups', metavar = 'GROUP', nargs = '+', default = ['dev'],
      help = "The group (i.e., 'dev' or  'eval') for which the models and scores should be generated")

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
      help = 'Normalize features before ISV training?')
  other_group.add_argument('-C', '--clean-intermediate', action='store_true',
      help = 'Clean up temporary files of older iterations?')

  skip_group.add_argument('--skip-normalization', '--non', action='store_true',
      help = "Skip the feature normalization step")
  skip_group.add_argument('--skip-k-means', '--nok', action='store_true',
      help = "Skip the KMeans step")
  skip_group.add_argument('--skip-gmm', '--nog', action='store_true',
      help = "Skip the GMM step")
  skip_group.add_argument('--skip-gmm-projection', '--nogp', action='store_true',
      help = "Skip the GMM projection step")
  skip_group.add_argument('--skip-isv', '--noi', action='store_true',
      help = "Skip the ISV step")
  skip_group.add_argument('--skip-isv-projection', '--noip', action='store_true',
      help = "Skip the GMM isv projection")

  #######################################################################################
  #################### sub-tasks being executed by this script ##########################
  parser.add_argument('--sub-task',
      choices = ('preprocess', 'train-extractor', 'extract', 'normalize-features', 'kmeans-init', 'kmeans-e-step', 'kmeans-m-step', 'gmm-init', 'gmm-e-step', 'gmm-m-step', 'gmm-project', 'train-isv', 'isv-project', 'enroll', 'compute-scores', 'concatenate'),
      help = argparse.SUPPRESS) #'Executes a subtask (FOR INTERNAL USE ONLY!!!)'
  parser.add_argument('--iteration', type=int,
      help = argparse.SUPPRESS) #'The current iteration of KMeans or GMM training'
  parser.add_argument('--model-type', choices = ['N', 'T'],
      help = argparse.SUPPRESS) #'Which type of models to generate (Normal or TModels)'
  parser.add_argument('--score-type', choices = ['A', 'B', 'C', 'D', 'Z'],
      help = argparse.SUPPRESS) #'The type of scores that should be computed'
  parser.add_argument('--group',
      help = argparse.SUPPRESS) #'The group for which the current action should be performed'

  return parser.parse_args(command_line_parameters)


def face_verify(args, command_line_parameters, external_dependencies = [], external_fake_job_id = 0):
  """This is the main entry point for computing face verification experiments.
  You just have to specify configuration scripts for any of the steps of the toolchain, which are:
  -- the database
  -- the preprocessing
  -- the feature extraction
  -- the score computation tool
  -- and the grid configuration (in case, the function should be executed in the grid).
  Additionally, you can skip parts of the toolchain by selecting proper --skip-... parameters.
  If your probe files are not too big, you can also specify the --preload-probes switch to speed up the score computation.
  If files should be re-generated, please specify the --force option (might be combined with the --skip-... options)."""


  # generate tool chain executor
  executor = ToolChainExecutorISV(args)
  # as the main entry point, check whether the grid option was given
  if args.sub_task:
    # execute the desired sub-task
    executor.execute_grid_job()
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


