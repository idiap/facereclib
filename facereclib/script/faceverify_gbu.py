#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
from __future__ import print_function

import sys, os
import argparse

from . import ToolChainExecutor
from .. import toolchain
from .. import utils

class ToolChainExecutorGBU (ToolChainExecutor.ToolChainExecutor):

  def __init__(self, args, protocol, perform_training):
    # call base class constructor
    ToolChainExecutor.ToolChainExecutor.__init__(self, args)

    # select the protocol
    self.m_database.protocol = protocol
    self.m_perform_training = perform_training

    if args.training_set:
      self.m_database.all_files_options.update({'subworld' : args.training_set})
      self.m_database.extractor_training_options.update({'subworld' : args.training_set})
      self.m_database.projector_training_options.update({'subworld' : args.training_set})
      self.m_database.enroller_training_options.update({'subworld' : args.training_set})


    # add specific configuration for ZT-normalization
    self.m_configuration.models_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.models_directory, self.m_database.protocol)

    self.m_configuration.scores_directory = os.path.join(self.m_configuration.user_directory, self.m_args.score_sub_directory, self.m_database.protocol, args.score_directory)

    # specify the file selector to be used
    self.m_file_selector = toolchain.FileSelector(
        self.m_database,
        preprocessed_directory = self.m_configuration.preprocessed_directory,
        extractor_file = self.m_configuration.extractor_file,
        features_directory = self.m_configuration.features_directory,
        projector_file = self.m_configuration.projector_file,
        projected_directory = self.m_configuration.projected_directory,
        enroller_file = self.m_configuration.enroller_file,
        model_directories = (self.m_configuration.models_directory,),
        score_directories = (self.m_configuration.scores_directory,)
    )

    # specify the file selector and tool chain objects to be used by this class (and its base class)
    self.m_tool_chain = toolchain.ToolChain(self.m_file_selector)


  def execute_tool_chain(self):
    """Executes the desired tool chain on the local machine"""
    utils.info("Executing face recognition algorithm on protocol '%s'" % self.m_database.protocol)
    # preprocessing
    if not self.m_args.skip_preprocessing:
      if self.m_args.dry_run:
        print ("Would have preprocessed data for protocol '%s' ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.preprocess_data(
              self.m_preprocessor,
              groups = self.groups(),
              force = self.m_args.force)

    # feature extraction
    if self.m_perform_training and not self.m_args.skip_extractor_training and self.m_extractor.requires_training:
      if self.m_args.dry_run:
        print ("Would have trained the extractor ...")
      else:
        self.m_tool_chain.train_extractor(
              self.m_extractor,
              self.m_preprocessor,
              force = self.m_args.force)

    if not self.m_args.skip_extraction:
      if self.m_args.dry_run:
        print ("Would have extracted the features for protocol '%s' ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.extract_features(
              self.m_extractor,
              self.m_preprocessor,
              groups = self.groups(),
              force = self.m_args.force)

    # feature projection
    if self.m_perform_training and not self.m_args.skip_projector_training and self.m_tool.requires_projector_training:
      if self.m_args.dry_run:
        print ("Would have trained the projector ...")
      else:
        self.m_tool_chain.train_projector(
              self.m_tool,
              self.m_extractor,
              force = self.m_args.force)

    if not self.m_args.skip_projection and self.m_tool.performs_projection:
      if self.m_args.dry_run:
        print ("Would have projected the features for protocol '%s' ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.project_features(
              self.m_tool,
              self.m_extractor,
              groups = self.groups(),
              force = self.m_args.force)

    # model enrollment
    if self.m_perform_training and not self.m_args.skip_enroller_training and self.m_tool.requires_enroller_training:
      if self.m_args.dry_run:
        print ("Would have trained the enroller ...")
      else:
        self.m_tool_chain.train_enroller(
              self.m_tool,
              self.m_extractor,
              force = self.m_args.force)

    if not self.m_args.skip_enrollment:
      if self.m_args.dry_run:
        print ("Would have enrolled the models for protocol '%s' ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.enroll_models(
              self.m_tool,
              self.m_extractor,
              compute_zt_norm = False,
              groups = ['dev'], # only dev group
              force = self.m_args.force)

    # score computation
    if not self.m_args.skip_score_computation:
      if self.m_args.dry_run:
        print ("Would have computed the scores for protocol '%s' ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.compute_scores(
              self.m_tool,
              compute_zt_norm = False,
              groups = ['dev'], # only dev group
              preload_probes = self.m_args.preload_probes,
              force = self.m_args.force)

    # concatenation of scores
    if not self.m_args.skip_concatenation:
      if self.m_args.dry_run:
        print ("Would have concatenated the scores for protocol '%s' ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.concatenate(
              compute_zt_norm = False,
              groups = ['dev']) # only dev group


  def add_jobs_to_grid(self, external_dependencies, external_job_ids):
    # collect job ids
    job_ids = {}
    job_ids.update(external_job_ids)

    # if there are any external dependencies, we need to respect them
    deps = external_dependencies[:]
    training_deps = external_dependencies[:]

    default_opt = ' --protocol %s'%self.m_database.protocol
    if self.m_perform_training:
      default_opt += ' --perform-training'
    # preprocessing; never has any dependencies.
    if not self.m_args.skip_preprocessing:
      # preprocessing must be done one after each other
      #   since training files are identical for all protocols
      preprocessing_deps = deps[:]
      if 'preprocessing' in job_ids:
        preprocessing_deps.append(job_ids['preprocessing'])
      job_ids['preprocessing'] = self.submit_grid_job(
              'preprocess' + default_opt,
              name = 'pre-%s' % self.m_database.protocol,
              number_of_parallel_jobs = self.m_grid.number_of_preprocessing_jobs,
              dependencies = preprocessing_deps,
              **self.m_grid.preprocessing_queue)
      deps.append(job_ids['preprocessing'])
      if self.m_perform_training:
        training_deps.append(job_ids['preprocessing'])

    # feature extraction training
    if self.m_perform_training and not self.m_args.skip_extractor_training and self.m_extractor.requires_training:
      job_ids['extraction_training'] = self.submit_grid_job(
              'train-extractor' + default_opt,
              name = 'f-train',
              dependencies = training_deps,
              **self.m_grid.training_queue)
    if 'extraction_training' in job_ids:
      deps.append(job_ids['extraction_training'])

    if not self.m_args.skip_extraction:
      job_ids['feature_extraction'] = self.submit_grid_job(
              'extract' + default_opt,
              name = 'extr-%s' % self.m_database.protocol,
              number_of_parallel_jobs = self.m_grid.number_of_extraction_jobs,
              dependencies = deps,
              **self.m_grid.extraction_queue)
      deps.append(job_ids['feature_extraction'])
      if self.m_perform_training:
        training_deps.append(job_ids['feature_extraction'])

    # feature projection training
    if self.m_perform_training and not self.m_args.skip_projector_training and self.m_tool.requires_projector_training:
      job_ids['projector_training'] = self.submit_grid_job(
              'train-projector' + default_opt,
              name = "p-train",
              dependencies = training_deps,
              **self.m_grid.training_queue)
    if 'projector_training' in job_ids:
      deps.append(job_ids['projector_training'])

    if not self.m_args.skip_projection and self.m_tool.performs_projection:
      job_ids['feature_projection'] = self.submit_grid_job(
              'project' + default_opt,
              name="pro-%s" % self.m_database.protocol,
              number_of_parallel_jobs = self.m_grid.number_of_projection_jobs,
              dependencies = deps,
              **self.m_grid.projection_queue)
      deps.append(job_ids['feature_projection'])
      if self.m_perform_training:
        training_deps.append(job_ids['feature_projection'])

    # model enrollment training
    if self.m_perform_training and not self.m_args.skip_enroller_training and self.m_tool.requires_enroller_training:
      job_ids['enrollment_training'] = self.submit_grid_job(
              'train-enroller' + default_opt,
              name="e-train",
              dependencies = training_deps,
              **self.m_grid.training_queue)
    if 'enrollment_training' in job_ids:
      deps.append(job_ids['enrollment_training'])

    # enroll models
    if not self.m_args.skip_enrollment:
      job_ids['enroll'] = self.submit_grid_job(
              'enroll' + default_opt,
              name = "enr-%s" % self.m_database.protocol,
              number_of_parallel_jobs = self.m_grid.number_of_enrollment_jobs,
              dependencies = deps,
              **self.m_grid.enrollment_queue)
      deps.append(job_ids['enroll'])

    # compute scores
    if not self.m_args.skip_score_computation:
      job_ids['score'] = self.submit_grid_job(
              'compute-scores' + default_opt,
              name = "score-%s" % self.m_database.protocol,
              number_of_parallel_jobs = self.m_grid.number_of_scoring_jobs,
              dependencies = deps,
              **self.m_grid.scoring_queue)
      deps.append(job_ids['score'])

    # concatenate results
    if not self.m_args.skip_concatenation:
      job_ids['concatenate'] = self.submit_grid_job(
              'concatenate' + default_opt,
              dependencies = deps,
              name = "concat-%s" % self.m_database.protocol)

    # return the job ids, in case anyone wants to know them
    return job_ids


  def execute_grid_job(self):
    """This function executes the grid job that is specified on the command line."""
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
      self.m_tool_chain.enroll_models(
          self.m_tool,
          self.m_extractor,
          indices = self.indices(self.m_file_selector.model_ids('dev'), self.m_grid.number_of_enrollment_jobs),
          compute_zt_norm = False,
          groups = ['dev'],
          force = self.m_args.force)

    # compute scores
    elif self.m_args.sub_task == 'compute-scores':
      self.m_tool_chain.compute_scores(
          self.m_tool,
          indices = self.indices(self.m_file_selector.model_ids('dev'), self.m_grid.number_of_scoring_jobs),
          compute_zt_norm = False,
          groups = ['dev'],
          preload_probes = self.m_args.preload_probes,
          force = self.m_args.force)

    # concatenate
    elif self.m_args.sub_task == 'concatenate':
      self.m_tool_chain.concatenate(
          compute_zt_norm = False,
          groups = ['dev'])

    # Test if the keyword was processed
    else:
      raise ValueError("The given subtask '%s' could not be processed. THIS IS A BUG. Please report this to the authors." % self.m_args.sub_task)


def parse_args(command_line_parameters):
  """This function parses the given options (which by default are the command line options)"""

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      conflict_handler='resolve')

  # add the arguments required for all tool chains
  config_group, dir_group, file_group, sub_dir_group, other_group, skip_group = ToolChainExecutorGBU.required_command_line_options(parser)

  # overwrite default database entry
  config_group.add_argument('-d', '--database', default = ['gbu'], nargs = '+',
      help = 'The database interface to be used. The default should work fine for the common cases.')

  sub_dir_group.add_argument('--models-directory', type = str, metavar = 'DIR', default = 'models',
      help = 'Subdirectories (of the --temp-directory) where the models should be stored')

  sub_dir_group.add_argument('--score-directory', metavar = 'DIR', default = 'nonorm',
      help = 'Sub-directory (of --user-directory) where to write the results to (used mainly to create directory structures consistent with the faceverify.py script)')

  #######################################################################################
  ############################ other options ############################################
  other_group.add_argument('-F', '--force', action='store_true',
      help = 'Force to erase former data if already exist')
  other_group.add_argument('-w', '--preload-probes', action='store_true', dest='preload_probes',
      help = 'Preload probe files during score computation (needs more memory, but is faster and requires fewer file accesses). WARNING! Use this flag with care!')
  other_group.add_argument('--protocols', type=str, nargs = '+', choices = ['Good', 'Bad', 'Ugly'], default = ['Good', 'Bad', 'Ugly'],
      help = 'The protocols to use, by default all three (Good, Bad, and Ugly) are executed.')
  other_group.add_argument('-x', '--training-set', choices=['x1', 'x2', 'x4', 'x8'],
      help = 'Select the training set to be used. Please do not use this option in a series of calls since this might influence other calls.')

  #######################################################################################
  #################### sub-tasks being executed by this script ##########################
  parser.add_argument('--sub-task',
      choices = ('preprocess', 'train-extractor', 'extract', 'train-projector', 'project', 'train-enroller', 'enroll', 'compute-scores', 'concatenate'),
      help = argparse.SUPPRESS) #'Executes a subtask (FOR INTERNAL USE ONLY!!!)'
  parser.add_argument('--protocol', type=str, choices=['Good','Bad','Ugly'],
      help = argparse.SUPPRESS) #'The protocol which should be used in this sub-task'
  parser.add_argument('--perform-training', action='store_true',
      help = argparse.SUPPRESS) #'Is this the first job that needs to perform the training?'

  #######################################################################################
  ####### shortcuts for the --skip-... commands #########################################
  skip_choices = ('preprocessing', 'extractor-training', 'extraction', 'projector-training', 'projection', 'enroller-training', 'enrollment', 'score-computation', 'concatenation')
  skip_group.add_argument('--execute-only', nargs = '+', choices = skip_choices,
      help = 'Executes only the given parts of the tool chain.')

  args = parser.parse_args(command_line_parameters)

  # set groups to be 'dev' only
  args.groups = ['dev',]

  if args.execute_only is not None:
    for skip in skip_choices:
      if skip not in args.execute_only:
        exec("args.skip_%s = True" % (skip.replace("-", "_")))
  return args


def face_verify(args, command_line_parameters, external_dependencies = [], external_fake_job_id = 0):
  """This is the main entry point for computing face verification experiments.
  You just have to specify configuration scripts for any of the steps of the tool chain, which are:
  -- the database
  -- the preprocessing
  -- feature extraction
  -- the recognition tool
  -- and the grid configuration (in case, the function should be executed in the grid).
  Additionally, you can skip parts of the tool chain by selecting proper --skip-... parameters.
  If your probe files are not too big, you can also specify the --preload-probes switch to speed up the score computation.
  If files should be re-generated, please specify the --force option (might be combined with the --skip-... options)"""

  if args.sub_task:
    # execute the desired sub-task
    executor = ToolChainExecutorGBU(args, args.protocol, args.perform_training)
    executor.execute_grid_job()
    return {}

  elif args.grid:

    # get the name of this file
    this_file = __file__
    if this_file[-1] == 'c':
      this_file = this_file[0:-1]

    # for the first protocol, we do not have any own dependencies
    dependencies = external_dependencies
    job_ids = {}
    resulting_dependencies = {}
    perform_training = True
    dry_run_init = external_fake_job_id
    for protocol in args.protocols:
      # create an executor object
      executor = ToolChainExecutorGBU(args, protocol, perform_training)
      # write the info file, but only for the first protocol
      if protocol == args.protocols[0]:
        executor.write_info(command_line_parameters)
      executor.set_common_parameters(calling_file = this_file, parameters = command_line_parameters, fake_job_id = dry_run_init)

      # add the jobs
      new_job_ids = executor.add_jobs_to_grid(dependencies, job_ids)
      job_ids.update(new_job_ids)

      # perform training only in the first round since the training set is identical for all algorithms
      perform_training = False

      dry_run_init += 30

    if executor.m_grid.is_local() and args.run_local_scheduler:
      if args.dry_run:
        print ("Would have started the local scheduler to finally run the experiments with parallel jobs")
      else:
        # start the jman local deamon
        executor.execute_local_deamon()
      return {}

    # at the end of all protocols, return the list of dependencies
    return job_ids
  else:
    perform_training = True
    # not in a grid, use default tool chain sequentially
    for protocol in args.protocols:
      # generate executor for the current protocol
      executor = ToolChainExecutorGBU(args, protocol, perform_training)
      executor.write_info(command_line_parameters)
      # execute the tool chain locally
      executor.execute_tool_chain()
      perform_training = False

    # no dependencies since we executed the jobs locally
    return {}


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
