#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

from __future__ import print_function

import os, sys, math
import argparse

from .. import toolchain
from .. import utils

class Configuration:
  """This class stores the basic configuration of the experiments.
  This configuration includes directories and files that are used."""

  def __init__(self, args, database_name, use_local_files):
    """Creates the default configuration based on the command line options."""
    # add command line based arguments
    if args.user_directory:
      self.user_directory = os.path.join(args.user_directory, args.sub_directory)
    else:
      self.user_directory = os.path.join("results", database_name, args.sub_directory)

    if args.temp_directory:
      self.temp_directory = os.path.join(args.temp_directory, args.sub_directory)
    else:
      self.temp_directory = os.path.join("temp", database_name, args.sub_directory)

    self.extractor_file = os.path.join(self.temp_directory, args.extractor_file)
    self.projector_file = os.path.join(self.temp_directory, args.projector_file)
    self.enroller_file = os.path.join(self.temp_directory, args.enroller_file)

    self.preprocessed_directory = os.path.join(self.temp_directory, args.preprocessed_data_directory)
    self.features_directory = os.path.join(self.temp_directory, args.features_directory)
    self.projected_directory = os.path.join(self.temp_directory, args.projected_features_directory)

    self.info_file = os.path.join(self.user_directory, "Experiment.info") if not args.experiment_info_file else args.experiment_info_file



class ToolChainExecutor:
  """This class is a helper class to provide functionality to execute tool chains.
  It manages the configuration files and the command line options, as well as the parallel execution of the tasks in the Idiap SGE grid."""

  def __init__(self, args):
    """Initializes the Tool chain executor."""
    # remember command line arguments
    self.m_args = args

    # generate the tools that we will need
    self.m_database = utils.resources.load_resource(' '.join(args.database), 'database', imports = args.imports)
    self.m_preprocessor = utils.resources.load_resource(' '.join(args.preprocessor), 'preprocessor', imports = args.imports)
    self.m_extractor = utils.resources.load_resource(' '.join(args.features), 'feature_extractor', imports = args.imports)
    self.m_tool = utils.resources.load_resource(' '.join(args.tool), 'tool', imports = args.imports)

    # load configuration files specified on command line
    use_local_files = True
    if args.grid:
      self.m_grid = utils.resources.load_resource(' '.join(args.grid), 'grid', imports = args.imports)
      use_local_files = self.m_grid.is_local()

    # generate configuration
    self.m_configuration = Configuration(args, self.m_database.name, use_local_files)

    utils.set_verbosity_level(args.verbose)

  def write_info(self, command_line_parameters):
    # write configuration
    try:
      utils.ensure_dir(os.path.dirname(self.m_configuration.info_file))
      f = open(self.m_configuration.info_file, 'w')
      f.write("Command line:\n")
      f.write(" ".join(command_line_parameters) + "\n\n")
      f.write("Configuration:\n")
      f.write("Database:\n%s\n\n" % self.m_database)
      f.write("Preprocessing:\n%s\n\n" % self.m_preprocessor)
      f.write("Feature Extraction:\n%s\n\n" % self.m_extractor)
      f.write("Algorithm:\n%s\n\n" % self.m_tool)
    except IOError:
      utils.error("Could not write the experimental setup into file '%s'" % self.m_configuration.info_file)

  def groups(self):
    """Checks the groups, for which the files must be preprocessed, and features must be extracted and projected."""
    groups = self.m_args.groups[:]
    if self.m_extractor.requires_training or self.m_tool.requires_projector_training or self.m_tool.requires_enroller_training:
      groups.append('world')
    return groups


  @staticmethod
  def required_command_line_options(parser, exclude_resources_from = []):
    """Initializes the minimum command line options that are required to run this experiment."""

    #######################################################################################
    ############## options that are required to be specified #######################
    config_group = parser.add_argument_group('\nParameters defining the experiment. Most of these parameters can be a registered resource, a configuration file, or even a string that defines a newly created object')
    config_group.add_argument('-d', '--database', metavar = 'x', nargs = '+', required = True,
        help = 'Database and the protocol; registered databases are: %s'%utils.resources.resource_keys('database', exclude_resources_from))
    config_group.add_argument('-p', '--preprocessing', metavar = 'x', nargs = '+', dest = 'preprocessor', required = True,
        help = 'Image preprocessing; registered preprocessors are: %s'%utils.resources.resource_keys('preprocessor', exclude_resources_from))
    config_group.add_argument('-f', '--features', metavar = 'x', nargs = '+', required = True,
        help = 'Feature extraction; registered feature extractors are: %s'%utils.resources.resource_keys('feature_extractor', exclude_resources_from))
    config_group.add_argument('-t', '--tool', metavar = 'x', nargs = '+', required = True,
        help = 'Face recognition; registered face recognition tools are: %s'%utils.resources.resource_keys('tool', exclude_resources_from))
    config_group.add_argument('-g', '--grid', metavar = 'x', nargs = '+',
        help = 'Configuration for the grid setup; if not specified, the commands are executed sequentially on the local machine.')
    config_group.add_argument('--imports', metavar = 'LIB', nargs = '+', default = ['facereclib'],
        help = 'If one of your configuration files is an actual command, please specify the lists of required imports to execute this command')
    config_group.add_argument('-b', '--sub-directory', metavar = 'DIR', required = True,
        help = 'The sub-directory where the files of the current experiment should be stored. Please specify a directory name with a name describing your experiment.')

    #######################################################################################
    ############## options to modify default directories or file names ####################
    dir_group = parser.add_argument_group('\nDirectories that can be changed according to your requirements')
    dir_group.add_argument('-T', '--temp-directory', metavar = 'DIR',
        help = 'The directory for temporary files; if not specified, "temp" in the current directory is used.')
    dir_group.add_argument('-U', '--result-directory', '--user-directory', metavar = 'DIR', dest='user_directory',
        help = 'The directory for resulting score files; if not specified, "results" in the current directory is used.')
    dir_group.add_argument('-s', '--score-sub-directory', metavar = 'DIR', default = 'scores',
        help = 'The sub-directory where to write the scores to.')

    file_group = parser.add_argument_group('\nName (maybe including a path relative to the --temp-directory) of files that will be generated. Note that not all files will be used by all tools.')
    file_group.add_argument('--extractor-file', metavar = 'FILE', default = 'Extractor.hdf5',
        help = 'Name of the file to write the feature extractor into.')
    file_group.add_argument('--projector-file', metavar = 'FILE', default = 'Projector.hdf5',
        help = 'Name of the file to write the feature projector into.')
    file_group.add_argument('--enroller-file' , metavar = 'FILE', default = 'Enroller.hdf5',
        help = 'Name of the file to write the model enroller into.')
    file_group.add_argument('-G', '--submit-db-file', metavar = 'FILE', default = 'submitted.sql3', dest = 'gridtk_database_file',
        help = 'The database file in which the submitted jobs will be written (only valid with the --grid option).')
    file_group.add_argument('--experiment-info-file', metavar = 'FILE',
        help = 'The file where the configuration of all parts of the experiments are written. If not specified, "Experiment.info" in the --result-directory is used.')

    sub_dir_group = parser.add_argument_group('\nSubdirectories of certain parts of the tool chain. You can specify directories in case you want to reuse parts of the experiments (e.g. extracted features) in other experiments. Please note that these directories are relative to the --temp-directory, but you can also specify absolute paths')
    sub_dir_group.add_argument('--preprocessed-data-directory', metavar = 'DIR', default = 'preprocessed',
        help = 'Name of the directory of the preprocessed data.')
    sub_dir_group.add_argument('--features-directory', metavar = 'DIR', default = 'features',
        help = 'Name of the directory of the features.')
    sub_dir_group.add_argument('--projected-features-directory', metavar = 'DIR', default = 'projected',
        help = 'Name of the directory where the projected data should be stored.')

    other_group = parser.add_argument_group('\nFlags that change the behavior of the experiment')
    other_group.add_argument('-q', '--dry-run', action='store_true',
        help = 'Only report the commands that will be executed, but do not execute them.')
    other_group.add_argument('-Z', '--write-compressed-score-files', action='store_true',
        help = 'Writes score files which are compressed with tar.bz2.')
    other_group.add_argument('-R', '--delete-dependent-jobs-on-failure', action='store_true',
        help = 'Try to recursively delete the dependent jobs from the SGE grid queue, when a job failed')
    other_group.add_argument('-X', '--external-dependencies', type=int, default = [], nargs='+',
        help = 'The jobs submitted to the grid have dependencies on the given job ids.')
    other_group.add_argument('-D', '--timer', choices=('real', 'system', 'user'), nargs = '*',
        help = 'Measure and report the time required by the execution of the tool chain (only on local machine)')
    other_group.add_argument('-L', '--run-local-scheduler', action='store_true',
        help = 'Starts the local scheduler after submitting the jobs to the local queue (by default, local jobs must be started by hand, e.g., using ./bin/jman --local -vv run-scheduler -x)')

    utils.add_logger_command_line_option(other_group)

    #######################################################################################
    ################# options for skipping parts of the toolchain #########################
    skip_group = parser.add_argument_group('\nFlags that allow to skip certain parts of the experiments. This does only make sense when the generated files are already there (e.g. when reusing parts of other experiments)')
    skip_group.add_argument('--skip-preprocessing', '--nopre', action='store_true',
        help = 'Skip the preprocessing step.')
    skip_group.add_argument('--skip-extractor-training', '--noet', action='store_true',
        help = 'Skip the feature extractor training step.')
    skip_group.add_argument('--skip-extraction', '--noe', action='store_true',
        help = 'Skip the feature extraction step.')
    skip_group.add_argument('--skip-projector-training', '--noprot', action='store_true',
        help = 'Skip the feature projector training step.')
    skip_group.add_argument('--skip-projection', '--nopro', action='store_true',
        help = 'Skip the feature projection step.')
    skip_group.add_argument('--skip-enroller-training', '--noenrt', action='store_true',
        help = 'Skip the model enroller training step.')
    skip_group.add_argument('--skip-enrollment', '--noenr', action='store_true',
        help = 'Skip the model enrollment step.')
    skip_group.add_argument('--skip-score-computation', '--nosc', action='store_true',
        help = 'Skip the score computation step.')
    skip_group.add_argument('--skip-concatenation', '--nocat', action='store_true',
        help = 'Skip the score concatenation step.')

    return (config_group, dir_group, file_group, sub_dir_group, other_group, skip_group)



################################################################################################
########################## Functions concerning the grid setup #################################
################################################################################################



  def set_common_parameters(self, calling_file, parameters, fake_job_id = 0, temp_dir = None):
    """Sets the parameters that the grid jobs require to be called.
    Just hand over all parameters of the faceverify script, and this function will do the rest.
    Please call this function before submitting jobs to the grid using the submit_jobs_to_grid function"""

    import gridtk
    # set gridtk logger to use the same output and format as we do
    utils.add_bob_handlers('gridtk')

    # we want to have the executable with the name of this file, which is laying in the bin directory
    self.m_common_parameters = [p for p in parameters[1:] if not '--skip' in p and not '--no' in p and p not in ('-q', '--dry-run')]

    # job id used for the dry-run
    self.m_fake_job_id = fake_job_id

    # define the dir from which the current executable was called
    #TODO: Find a more clever way to get the directory, where the script is installed.
    if os.path.exists(parameters[0]):
      self.m_bin_directory = os.path.dirname(os.path.realpath(parameters[0]))
    else:
      # This should happen only during nose testing under some weird conditions.
      # Since nose tests should not actually run anything in the grid, we can use a fake directory here.
      self.m_bin_directory = './bin'
    self.m_executable = os.path.join(self.m_bin_directory, os.path.basename(calling_file))
    self.m_jman = os.path.join(self.m_bin_directory, 'jman')
    # generate job manager and set the temp dir
    if self.m_grid.grid_type == 'local':
      self.m_job_manager = gridtk.local.JobManagerLocal(database = self.m_args.gridtk_database_file, wrapper_script = self.m_jman)
    elif self.m_grid.grid_type == 'sge':
      self.m_job_manager = gridtk.sge.JobManagerSGE(database = self.m_args.gridtk_database_file, wrapper_script = self.m_jman)
    else:
      raise ValueError("The JobManager type '%s' is not supported.")
    self.m_logs_directory = os.path.join(temp_dir if temp_dir else self.m_configuration.temp_directory, "grid_tk_logs")
    self.m_submitted_job_ids = []


  def indices(self, list_to_split, number_of_parallel_jobs, task_id=None):
    """This function returns the first and last index for the files for the current job ID.
       If no job id is set (e.g., because a sub-job is executed locally), it simply returns all indices."""
    # test if the 'SEG_TASK_ID' environment is set
    sge_task_id = os.getenv('SGE_TASK_ID') if task_id is None else task_id
    if sge_task_id is None:
      # task id is not set, so this function is not called from a grid job
      # hence, we process the whole list
      return (0,len(list_to_split))
    else:
      job_id = int(sge_task_id) - 1
      # compute number of files to be executed
      number_of_objects_per_job = int(math.ceil(float(len(list_to_split) / float(number_of_parallel_jobs))))
      start = job_id * number_of_objects_per_job
      end = min((job_id + 1) * number_of_objects_per_job, len(list_to_split))
      return (start, end)


  def submit_grid_job(self, command, number_of_parallel_jobs = 1, dependencies=[], name = None, **kwargs):
    """Submits a job to the grid."""

    dependencies = dependencies[:] + self.m_args.external_dependencies
    # create the command to be executed
    cmd = [
            self.m_executable,
            '--sub-task'
          ]
    cmd += command.split()
    cmd += self.m_common_parameters

    # if no job name is specified, create one
    if name is None:
      name = command.split(' ')[0]
      log_sub_dir = command.replace(' ','__')
    else:
      log_sub_dir = name
    # generate log directory
    logdir = os.path.join(self.m_logs_directory, log_sub_dir)

    # generate job array
    if number_of_parallel_jobs > 1:
      array = (1,number_of_parallel_jobs,1)
    else:
      array = None

    # submit the job to the job manager
    if not self.m_args.dry_run:
      job_id = self.m_job_manager.submit(
          command_line = cmd,
          name = name,
          array = array,
          dependencies = dependencies,
          log_dir = logdir,
          stop_on_failure = self.m_args.delete_dependent_jobs_on_failure,
          **kwargs
      )
      utils.info("submitted: job '%s' with id '%d' and dependencies '%s'" % (name, job_id, dependencies))
      self.m_submitted_job_ids.append(job_id)
      return job_id
    else:
      self.m_fake_job_id += 1
      print ('would have submitted job', name, 'with id', self.m_fake_job_id, 'with parameters', kwargs, end='')
      if array:
        print (' using', array[1], 'parallel jobs', end='')
      print (' as:', utils.command_line(cmd), '\nwith dependencies', dependencies)
      return self.m_fake_job_id


  def grid_job_id(self):
    id = os.getenv('JOB_ID')
    if id is not None:
      return int(id)
    return id

  def execute_local_deamon(self):
    """Starts the local deamon and waits until it has finished."""
    utils.info("Starting jman deamon to finally run the jobs on the local machine.")
    self.m_job_manager.run_scheduler(job_ids = self.m_submitted_job_ids, parallel_jobs=self.m_grid.number_of_parallel_processes, sleep_time=self.m_grid.scheduler_sleep_time, die_when_finished=True, nice=10)
