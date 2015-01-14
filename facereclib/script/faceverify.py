#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
from __future__ import print_function

import sys, os
import argparse

from . import ToolChainExecutor
from .. import toolchain, utils

class ToolChainExecutorZT (ToolChainExecutor.ToolChainExecutor):
  """Class that executes the ZT tool chain (locally or in the grid)."""

  def __init__(self, args):
    # call base class constructor
    ToolChainExecutor.ToolChainExecutor.__init__(self, args)

    # overwrite protocol from command line?
    if args.protocol:
      self.m_database.protocol = args.protocol

    protocol_subdir = self.m_database.protocol if self.m_database.protocol else "."

    self.m_configuration.models_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.models_directories[0], protocol_subdir)
    self.m_configuration.scores_no_norm_directory = os.path.join(self.m_configuration.user_directory, self.m_args.score_sub_directory, protocol_subdir, self.m_args.zt_score_directories[0])
    # add specific configuration for ZT-normalization
    if args.zt_norm:
      self.m_configuration.t_norm_models_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.models_directories[1], protocol_subdir)
      models_directories = (self.m_configuration.models_directory, self.m_configuration.t_norm_models_directory)

      self.m_configuration.scores_zt_norm_directory = os.path.join(self.m_configuration.user_directory, self.m_args.score_sub_directory, protocol_subdir, self.m_args.zt_score_directories[1])
      score_directories = (self.m_configuration.scores_no_norm_directory, self.m_configuration.scores_zt_norm_directory)

      self.m_configuration.zt_norm_A_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.score_sub_directory, protocol_subdir, self.m_args.zt_temp_directories[0])
      self.m_configuration.zt_norm_B_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.score_sub_directory, protocol_subdir, self.m_args.zt_temp_directories[1])
      self.m_configuration.zt_norm_C_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.score_sub_directory, protocol_subdir, self.m_args.zt_temp_directories[2])
      self.m_configuration.zt_norm_D_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.score_sub_directory, protocol_subdir, self.m_args.zt_temp_directories[3])
      self.m_configuration.zt_norm_D_sameValue_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.score_sub_directory, protocol_subdir, self.m_args.zt_temp_directories[4])
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
    self.m_tool_chain = toolchain.ToolChain(self.m_file_selector, self.m_args.write_compressed_score_files)


  def execute_tool_chain(self):
    """Executes the ZT tool chain on the local machine."""
    # preprocessing
    if not self.m_args.skip_preprocessing:
      if self.m_args.dry_run:
        print ("Would have preprocessed data ...")
      else:
        self.m_tool_chain.preprocess_data(
              self.m_preprocessor,
              groups = self.groups(),
              force = self.m_args.force)

    # feature extraction
    if not self.m_args.skip_extractor_training and self.m_extractor.requires_training:
      if self.m_args.dry_run:
        print ("Would have trained the extractor ...")
      else:
        self.m_tool_chain.train_extractor(
              self.m_extractor,
              self.m_preprocessor,
              force = self.m_args.force)

    if not self.m_args.skip_extraction:
      if self.m_args.dry_run:
        print ("Would have extracted the features ...")
      else:
        self.m_tool_chain.extract_features(
              self.m_extractor,
              self.m_preprocessor,
              groups = self.groups(),
              force = self.m_args.force)

    # feature projection
    if not self.m_args.skip_projector_training and self.m_tool.requires_projector_training:
      if self.m_args.dry_run:
        print ("Would have trained the projector ...")
      else:
        self.m_tool_chain.train_projector(
              self.m_tool,
              self.m_extractor,
              force = self.m_args.force)

    if not self.m_args.skip_projection and self.m_tool.performs_projection:
      if self.m_args.dry_run:
        print ("Would have projected the features ...")
      else:
        self.m_tool_chain.project_features(
              self.m_tool,
              self.m_extractor,
              groups = self.groups(),
              force = self.m_args.force)

    # model enrollment
    if not self.m_args.skip_enroller_training and self.m_tool.requires_enroller_training:
      if self.m_args.dry_run:
        print ("Would have trained the enroller ...")
      else:
        self.m_tool_chain.train_enroller(
              self.m_tool,
              self.m_extractor,
              force = self.m_args.force)

    if not self.m_args.skip_enrollment:
      if self.m_args.dry_run:
        print ("Would have enrolled the models of groups %s ..." % self.m_args.groups)
      else:
        self.m_tool_chain.enroll_models(
              self.m_tool,
              self.m_extractor,
              self.m_args.zt_norm,
              groups = self.m_args.groups,
              force = self.m_args.force)

    # score computation
    if not self.m_args.skip_score_computation:
      if self.m_args.dry_run:
        print ("Would have computed the scores of groups %s ..." % self.m_args.groups)
      else:
        self.m_tool_chain.compute_scores(
              self.m_tool,
              self.m_args.zt_norm,
              groups = self.m_args.groups,
              preload_probes = self.m_args.preload_probes,
              force = self.m_args.force)

      if self.m_args.zt_norm:
        if self.m_args.dry_run:
          print ("Would have computed the ZT-norm scores of groups %s ..." % self.m_args.groups)
        else:
          self.m_tool_chain.zt_norm(groups = self.m_args.groups)

    # concatenation of scores
    if not self.m_args.skip_concatenation:
      if self.m_args.dry_run:
        print ("Would have concatenated the scores of groups %s ..." % self.m_args.groups)
      else:
        self.m_tool_chain.concatenate(
              self.m_args.zt_norm,
              groups = self.m_args.groups)

    # calibration of scores
    if self.m_args.calibrate_scores:
      if self.m_args.dry_run:
        print ("Would have calibrated the scores of groups %s ..." % self.m_args.groups)
      else:
        self.m_tool_chain.calibrate_scores(
            norms = ['nonorm', 'ztnorm'] if self.m_args.zt_norm else ['nonorm'],
            groups = self.m_args.groups)



  def add_jobs_to_grid(self, external_dependencies):
    """Adds all (desired) jobs of the tool chain to the grid."""
    # collect the job ids
    job_ids = {}

    # if there are any external dependencies, we need to respect them
    deps = external_dependencies[:]

    # preprocessing
    if not self.m_args.skip_preprocessing:
      job_ids['preprocessing'] = self.submit_grid_job(
              'preprocess',
              number_of_parallel_jobs = self.m_grid.number_of_preprocessing_jobs,
              dependencies = deps,
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
              number_of_parallel_jobs = self.m_grid.number_of_extraction_jobs,
              dependencies = deps,
              **self.m_grid.extraction_queue)
      deps.append(job_ids['extraction'])

    # feature projection training
    if not self.m_args.skip_projector_training and self.m_tool.requires_projector_training:
      job_ids['projector_training'] = self.submit_grid_job(
              'train-projector',
              name="train-p",
              dependencies = deps,
              **self.m_grid.training_queue)
      deps.append(job_ids['projector_training'])

    # feature projection
    if not self.m_args.skip_projection and self.m_tool.performs_projection:
      job_ids['projection'] = self.submit_grid_job(
              'project',
              number_of_parallel_jobs = self.m_grid.number_of_projection_jobs,
              dependencies = deps,
              **self.m_grid.projection_queue)
      deps.append(job_ids['projection'])

    # model enrollment training
    if not self.m_args.skip_enroller_training and self.m_tool.requires_enroller_training:
      job_ids['enroller_training'] = self.submit_grid_job(
              'train-enroller',
              name = "train-e",
              dependencies = deps,
              **self.m_grid.training_queue)
      deps.append(job_ids['enroller_training'])

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
                number_of_parallel_jobs = self.m_grid.number_of_enrollment_jobs,
                dependencies = deps,
                **self.m_grid.enrollment_queue)
        enroll_deps_n[group].append(job_ids['enroll_%s_N'%group])

        if self.m_args.zt_norm:
          job_ids['enroll_%s_T'%group] = self.submit_grid_job(
                  'enroll --group %s --model-type T'%group,
                  name = "enr-T-%s"%group,
                  number_of_parallel_jobs = self.m_grid.number_of_enrollment_jobs,
                  dependencies = deps,
                  **self.m_grid.enrollment_queue)
          enroll_deps_t[group].append(job_ids['enroll_%s_T'%group])

      # compute A,B,C, and D scores
      if not self.m_args.skip_score_computation:
        job_ids['score_%s_A'%group] = self.submit_grid_job(
                'compute-scores --group %s --score-type A'%group,
                name = "score-A-%s"%group,
                number_of_parallel_jobs = self.m_grid.number_of_scoring_jobs,
                dependencies = enroll_deps_n[group],
                **self.m_grid.scoring_queue)
        concat_deps[group] = [job_ids['score_%s_A'%group]]

        if self.m_args.zt_norm:
          job_ids['score_%s_B'%group] = self.submit_grid_job(
                  'compute-scores --group %s --score-type B'%group,
                  name = "score-B-%s"%group,
                  number_of_parallel_jobs = self.m_grid.number_of_scoring_jobs,
                  dependencies = enroll_deps_n[group],
                  **self.m_grid.scoring_queue)

          job_ids['score_%s_C'%group] = self.submit_grid_job(
                  'compute-scores --group %s --score-type C'%group,
                  name = "score-C-%s"%group,
                  number_of_parallel_jobs = self.m_grid.number_of_scoring_jobs,
                  dependencies = enroll_deps_t[group],
                  **self.m_grid.scoring_queue)

          job_ids['score_%s_D'%group] = self.submit_grid_job(
                  'compute-scores --group %s --score-type D'%group,
                  name = "score-D-%s"%group,
                  number_of_parallel_jobs = self.m_grid.number_of_scoring_jobs,
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

    # calibrate the scores
    if self.m_args.calibrate_scores:
      calib_deps = [job_ids['concat_%s'%g] for g in self.m_args.groups if 'concat_%s'%g in job_ids]
      job_ids['calibrate'] = self.submit_grid_job(
              'calibrate',
              dependencies = calib_deps)


    # return the job ids, in case anyone wants to know them
    return job_ids


  def execute_grid_job(self):
    """Run the desired job of the ZT tool chain that is specified on command line."""
    # preprocess the data
    if self.m_args.sub_task == 'preprocess':
      self.m_tool_chain.preprocess_data(
          self.m_preprocessor,
          groups = self.groups(),
          indices = self.indices(self.m_file_selector.original_data_list(groups=self.groups()), self.m_grid.number_of_preprocessing_jobs),
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
          groups = self.groups(),
          indices = self.indices(self.m_file_selector.preprocessed_data_list(groups=self.groups()), self.m_grid.number_of_extraction_jobs),
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'train-projector':
      self.m_tool_chain.train_projector(
          self.m_tool,
          self.m_extractor,
          force = self.m_args.force)

    # project the features
    elif self.m_args.sub_task == 'project':
      self.m_tool_chain.project_features(
          self.m_tool,
          self.m_extractor,
          groups = self.groups(),
          indices = self.indices(self.m_file_selector.preprocessed_data_list(groups=self.groups()), self.m_grid.number_of_projection_jobs),
          force = self.m_args.force)

    # train the model enroller
    elif self.m_args.sub_task == 'train-enroller':
      self.m_tool_chain.train_enroller(
          self.m_tool,
          self.m_extractor,
          force = self.m_args.force)

    # enroll the models
    elif self.m_args.sub_task == 'enroll':
      if self.m_args.model_type == 'N':
        self.m_tool_chain.enroll_models(
            self.m_tool,
            self.m_extractor,
            self.m_args.zt_norm,
            indices = self.indices(self.m_file_selector.model_ids(self.m_args.group), self.m_grid.number_of_enrollment_jobs),
            groups = [self.m_args.group],
            types = ['N'],
            force = self.m_args.force)

      else:
        self.m_tool_chain.enroll_models(
            self.m_tool,
            self.m_extractor,
            self.m_args.zt_norm,
            indices = self.indices(self.m_file_selector.t_model_ids(self.m_args.group), self.m_grid.number_of_enrollment_jobs),
            groups = [self.m_args.group],
            types = ['T'],
            force = self.m_args.force)

    # compute scores
    elif self.m_args.sub_task == 'compute-scores':
      if self.m_args.score_type in ['A', 'B']:
        self.m_tool_chain.compute_scores(
            self.m_tool,
            self.m_args.zt_norm,
            indices = self.indices(self.m_file_selector.model_ids(self.m_args.group), self.m_grid.number_of_scoring_jobs),
            groups = [self.m_args.group],
            types = [self.m_args.score_type],
            preload_probes = self.m_args.preload_probes,
            force = self.m_args.force)

      elif self.m_args.score_type in ['C', 'D']:
        self.m_tool_chain.compute_scores(
            self.m_tool,
            self.m_args.zt_norm,
            indices = self.indices(self.m_file_selector.t_model_ids(self.m_args.group), self.m_grid.number_of_scoring_jobs),
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

    # calibrate scores
    elif self.m_args.sub_task == 'calibrate':
      self.m_tool_chain.calibrate_scores(
          norms = ['nonorm', 'ztnorm'] if self.m_args.zt_norm else ['nonorm'],
          groups = self.m_args.groups)

    # Test if the keyword was processed
    else:
      raise ValueError("The given subtask '%s' could not be processed. THIS IS A BUG. Please report this to the authors." % self.m_args.sub_task)


def parse_args(command_line_parameters, exclude_resources_from = []):
  """This function parses the given options (which by default are the command line options). If exclude_resources_from is specified (as a list), the resources from the given packages are not listed in the help message."""
  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # add the arguments required for all tool chains
  config_group, dir_group, file_group, sub_dir_group, other_group, skip_group = ToolChainExecutorZT.required_command_line_options(parser, exclude_resources_from)

  config_group.add_argument('-P', '--protocol', metavar='PROTOCOL',
      help = 'Overwrite the protocol that is stored in the database by the given one (might not by applicable for all databases).')

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
  other_group.add_argument('-c', '--calibrate-scores', action='store_true',
      help = 'Performs score calibration after the scores are computed.')
  other_group.add_argument('-F', '--force', action='store_true',
      help = 'Force to erase former data if already exist')
  other_group.add_argument('-w', '--preload-probes', action='store_true',
      help = 'Preload probe files during score computation (needs more memory, but is faster and requires fewer file accesses). WARNING! Use this flag with care!')
  other_group.add_argument('--groups', metavar = 'GROUP', nargs = '+', default = ['dev'],
      help = "The group (i.e., 'dev' or  'eval') for which the models and scores should be generated")

  #######################################################################################
  #################### sub-tasks being executed by this script ##########################
  parser.add_argument('--sub-task',
      choices = ('preprocess', 'train-extractor', 'extract', 'train-projector', 'project', 'train-enroller', 'enroll', 'compute-scores', 'concatenate', 'calibrate'),
      help = argparse.SUPPRESS) #'Executes a subtask (FOR INTERNAL USE ONLY!!!)'
  parser.add_argument('--model-type', choices = ['N', 'T'],
      help = argparse.SUPPRESS) #'Which type of models to generate (Normal or TModels)'
  parser.add_argument('--score-type', choices = ['A', 'B', 'C', 'D', 'Z'],
      help = argparse.SUPPRESS) #'The type of scores that should be computed'
  parser.add_argument('--group',
      help = argparse.SUPPRESS) #'The group for which the current action should be performed'

  #######################################################################################
  ####### shortcuts for the --skip-... commands #########################################
  skip_choices = ('preprocessing', 'extractor-training', 'extraction', 'projector-training', 'projection', 'enroller-training', 'enrollment', 'score-computation', 'concatenation')
  skip_group.add_argument('--execute-only', nargs = '+', choices = skip_choices,
      help = 'Executes only the given parts of the tool chain.')

  args = parser.parse_args(command_line_parameters)

  if args.execute_only is not None:
    for skip in skip_choices:
      if skip not in args.execute_only:
        exec("args.skip_%s = True" % (skip.replace("-", "_")))
  return args


def face_verify(args, command_line_parameters, external_dependencies = [], external_fake_job_id = 0):
  """This is the main entry point for computing face verification experiments.
  You just have to specify configurations for any of the steps of the toolchain, which are:
  -- the database
  -- the preprocessing
  -- feature extraction
  -- the recognition tool
  -- and the grid configuration (in case, the function should be executed in the grid).
  Additionally, you can skip parts of the toolchain by selecting proper --skip-... parameters.
  If your probe files are not too big, you can also specify the --preload-probes switch to speed up the score computation.
  If files should be re-generated, please specify the --force option (might be combined with the --skip-... options)."""


  # generate tool chain executor
  executor = ToolChainExecutorZT(args)
  # as the main entry point, check whether the sub-task is specified
  if args.sub_task is not None:
    # execute the desired sub-task
    executor.execute_grid_job()
    return {}
  elif not args.grid:
    if args.timer is not None and not len(args.timer):
      args.timer = ('real', 'system', 'user')
    # not in a grid, use default tool chain sequentially
    if args.timer:
      utils.info("- Timer: Starting timer")
      start_time = os.times()

    executor.write_info(command_line_parameters)

    executor.execute_tool_chain()

    if args.timer:
      end_time = os.times()
      utils.info("- Timer: Stopped timer")

      for t in args.timer:
        index = {'real':4, 'system':1, 'user':0}[t]
        print ("Elapsed", t ,"time:", end_time[index] - start_time[index], "seconds")

    return {}

  else:
    # no other parameter given, so deploy new jobs

    # get the name of this file
    this_file = __file__
    if this_file[-1] == 'c':
      this_file = this_file[0:-1]

    executor.write_info(command_line_parameters)

    # initialize the executor to submit the jobs to the grid
    executor.set_common_parameters(calling_file = this_file, parameters = command_line_parameters, fake_job_id = external_fake_job_id)

    # add the jobs
    job_ids = executor.add_jobs_to_grid(external_dependencies)

    if executor.m_grid.is_local() and args.run_local_scheduler:
      if args.dry_run:
        print ("Would have started the local scheduler to finally run the experiments with parallel jobs")
      else:
        # start the jman local deamon
        executor.execute_local_deamon()
      return {}

    else:
      return job_ids



def main(command_line_parameters = sys.argv):
  """Executes the main function"""
  try:
    # do the command line parsing
    args = parse_args(command_line_parameters[1:])

    # perform face verification test
    face_verify(args, command_line_parameters)
  except Exception as e:
    # track any exceptions as error logs (i.e., to get a time stamp)
    utils.error("During the execution, an exception was raised: %s" % e)
    raise

if __name__ == "__main__":
  main()
