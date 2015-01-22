#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
from __future__ import print_function

import bob.measure

import sys, os
import argparse
import numpy

from . import ToolChainExecutor
from .. import toolchain
from .. import utils

class ToolChainExecutorLFW (ToolChainExecutor.ToolChainExecutor):

  def __init__(self, args, protocol):
    # call base class constructor
    ToolChainExecutor.ToolChainExecutor.__init__(self, args)

    # overwrite the protocol of the database with the given one
    self.m_database.protocol = protocol

    # add specific configuration for LFW database
    # each fold might have its own feature extraction training and feature projection training,
    # so we have to overwrite the default directories
    view = 'view1' if protocol == 'view1' else 'view2'
    self.m_configuration.preprocessed_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.preprocessed_data_directory, view)
    self.m_configuration.features_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.features_directory, protocol)
    self.m_configuration.projected_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.projected_features_directory, protocol)

    self.m_configuration.extractor_file = os.path.join(self.m_configuration.temp_directory, protocol, self.m_args.extractor_file)
    self.m_configuration.projector_file = os.path.join(self.m_configuration.temp_directory, protocol, self.m_args.projector_file)
    self.m_configuration.enroller_file = os.path.join(self.m_configuration.temp_directory, protocol, self.m_args.enroller_file)

    self.m_configuration.models_directory = os.path.join(self.m_configuration.temp_directory, self.m_args.models_directory, protocol)
    self.m_configuration.scores_directory = self.__scores_directory__(protocol)

    # define the final result text file
    if self.m_args.result_file:
      self.m_configuration.result_file = self.m_args.result_file
    else:
      self.m_configuration.result_file = os.path.join(self.m_configuration.user_directory, self.m_args.score_sub_directory, 'results.txt')

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

    # create the tool chain to be used to actually perform the parts of the experiments
    self.m_tool_chain = toolchain.ToolChain(self.m_file_selector)


  def __scores_directory__(self, protocol):
    """This helper function returns the score directory for the given protocol."""
    return os.path.join(self.m_configuration.user_directory, self.m_args.score_sub_directory, protocol, self.m_args.score_directory)

  def execute_tool_chain(self):
    """Executes the desired tool chain on the local machine"""
    # preprocessing
    if not self.m_args.skip_preprocessing:
      if self.m_args.dry_run:
        print ("Would have preprocessed data for protocol %s ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.preprocess_data(
              self.m_preprocessor,
              groups = self.groups(),
              force = self.m_args.force)

    # feature extraction
    if not self.m_args.skip_extractor_training and self.m_extractor.requires_training:
      if self.m_args.dry_run:
        print ("Would have trained the extractor for protocol %s ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.train_extractor(
              self.m_extractor,
              self.m_preprocessor,
              force = self.m_args.force)

    if not self.m_args.skip_extraction:
      if self.m_args.dry_run:
        print ("Would have extracted the features for protocol %s ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.extract_features(
              self.m_extractor,
              self.m_preprocessor,
              groups = self.groups(),
              force = self.m_args.force)

    # feature projection
    if not self.m_args.skip_projector_training and self.m_tool.requires_projector_training:
      if self.m_args.dry_run:
        print ("Would have trained the projector for protocol %s ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.train_projector(
              self.m_tool,
              self.m_extractor,
              force = self.m_args.force)

    if not self.m_args.skip_projection and self.m_tool.performs_projection:
      if self.m_args.dry_run:
        print ("Would have projected the features for protocol %s ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.project_features(
              self.m_tool,
              self.m_extractor,
              groups = self.groups(),
              force = self.m_args.force)

    # model enrollment
    if not self.m_args.skip_enroller_training and self.m_tool.requires_enroller_training:
      if self.m_args.dry_run:
        print ("Would have trained the enroller for protocol %s ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.train_enroller(
              self.m_tool,
              self.m_extractor,
              force = self.m_args.force)

    if not self.m_args.skip_enrollment:
      if self.m_args.dry_run:
        print ("Would have enrolled the models for protocol %s ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.enroll_models(
              self.m_tool,
              self.m_extractor,
              groups = self.m_args.groups,
              compute_zt_norm = False,
              force = self.m_args.force)

    # score computation
    if not self.m_args.skip_score_computation:
      if self.m_args.dry_run:
        print ("Would have computed the scores for protocol %s ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.compute_scores(
              self.m_tool,
              compute_zt_norm = False,
              groups = self.m_args.groups,
              preload_probes = self.m_args.preload_probes,
              force = self.m_args.force)

    if not self.m_args.skip_concatenation:
      if self.m_args.dry_run:
        print ("Would have concatenated the scores for protocol %s ..." % self.m_database.protocol)
      else:
        self.m_tool_chain.concatenate(compute_zt_norm = False)

    # That's it. The final averaging of results will be done by the calling function


  def add_jobs_to_grid(self, external_dependencies):
    # collect job ids
    job_ids = {}

    # if there are any external dependencies, we need to respect them
    deps = external_dependencies[:]

    protocol = self.m_database.protocol
    pshort = protocol[0] + str(int(protocol[4:]) % 10)
    default_opt = ' --protocol %s'%protocol
    # preprocessing; never has any dependencies.
    if not self.m_args.skip_preprocessing:
      job_ids['preprocessing'] = self.submit_grid_job(
              'preprocess' + default_opt,
              name = 'pre-%s'%pshort,
              number_of_parallel_jobs = self.m_grid.number_of_preprocessing_jobs,
              dependencies = deps,
              **self.m_grid.preprocessing_queue)
      deps.append(job_ids['preprocessing'])

    # feature extraction training
    if not self.m_args.skip_extractor_training and self.m_extractor.requires_training:
      job_ids['extraction_training'] = self.submit_grid_job(
              'train-extractor' + default_opt,
              name = 'f-train-%s'%pshort,
              dependencies = deps,
              **self.m_grid.training_queue)
      deps.append(job_ids['extraction_training'])

    if not self.m_args.skip_extraction:
      job_ids['feature_extraction'] = self.submit_grid_job(
              'extract' + default_opt,
              name = 'extr-%s'%pshort,
              number_of_parallel_jobs = self.m_grid.number_of_extraction_jobs,
              dependencies = deps,
              **self.m_grid.extraction_queue)
      deps.append(job_ids['feature_extraction'])

    # feature projection training
    if not self.m_args.skip_projector_training and self.m_tool.requires_projector_training:
      job_ids['projector_training'] = self.submit_grid_job(
              'train-projector' + default_opt,
              name = "p-train-%s"%pshort,
              dependencies = deps,
              **self.m_grid.training_queue)
      deps.append(job_ids['projector_training'])

    if not self.m_args.skip_projection and self.m_tool.performs_projection:
      job_ids['feature_projection'] = self.submit_grid_job(
              'project' + default_opt,
              name="pro-%s"%pshort,
              number_of_parallel_jobs = self.m_grid.number_of_projection_jobs,
              dependencies = deps,
              **self.m_grid.projection_queue)
      deps.append(job_ids['feature_projection'])

    # model enrollment training
    if not self.m_args.skip_enroller_training and self.m_tool.requires_enroller_training:
      job_ids['enrollment_training'] = self.submit_grid_job(
              'train-enroller' + default_opt,
              dependencies = deps,
              name="e-train-%s"%pshort,
              **self.m_grid.training_queue)
      deps.append(job_ids['enrollment_training'])

    # enroll models
    groups = ['dev'] if protocol=='view1' else self.m_args.groups
    if not self.m_args.skip_enrollment:
      for group in groups:
        job_ids['enroll-%s'%group] = self.submit_grid_job(
                'enroll --group %s'%group + default_opt,
                name = "enr-%s-%s"%(pshort,group),
                number_of_parallel_jobs = self.m_grid.number_of_enrollment_jobs,
                dependencies = deps,
                **self.m_grid.enrollment_queue)
      for group in groups:
        deps.append(job_ids['enroll-%s'%group])

    # compute scores
    if not self.m_args.skip_score_computation:
      for group in groups:
        job_ids['score-%s'%group] = self.submit_grid_job(
                'compute-scores --group %s'%group + default_opt,
                number_of_parallel_jobs = self.m_grid.number_of_scoring_jobs,
                dependencies = deps,
                name = "score-%s-%s"%(pshort,group),
                **self.m_grid.scoring_queue)
      for group in groups:
        deps.append(job_ids['score-%s'%group])

    # concatenate results
    if not self.m_args.skip_concatenation:
      job_ids['concatenate'] = self.submit_grid_job(
              'concatenate' + default_opt,
              dependencies = deps,
              name = "concat-%s"%pshort)

    # return the job ids, in case anyone wants to know them
    return job_ids


  def add_average_job_to_grid(self, external_dependencies):
    """Adds the job to average the results of the runs"""
    return {'average' :
        self.submit_grid_job(
              'average-results --protocol view1',  # The protocol is ignored by this function, but has to be specified.
              dependencies = external_dependencies,
              name = "average")}


  def execute_grid_job(self):
    """This function executes the grid job that is specified on the command line."""
    # preprocess
    if self.m_args.sub_task == 'preprocess':
      self.m_tool_chain.preprocess_data(
          self.m_preprocessor,
          groups = self.groups(),
          indices = self.indices(self.m_file_selector.original_data_list(groups=self.groups()), self.m_grid.number_of_preprocessing_jobs),
          force = self.m_args.force)

    elif self.m_args.sub_task == 'train-extractor':
      self.m_tool_chain.train_extractor(
          self.m_extractor,
          self.m_preprocessor,
          force = self.m_args.force)

    # extract features
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

    # train model enroller
    elif self.m_args.sub_task == 'train-enroller':
      self.m_tool_chain.train_enroller(
          self.m_tool,
          self.m_extractor,
          force = self.m_args.force)

    # enroll models
    elif self.m_args.sub_task == 'enroll':
      self.m_tool_chain.enroll_models(
          self.m_tool,
          self.m_extractor,
          groups = (self.m_args.group,),
          compute_zt_norm = False,
          indices = self.indices(self.m_file_selector.model_ids(self.m_args.group), self.m_grid.number_of_enrollment_jobs),
          force = self.m_args.force)

    # compute scores
    elif self.m_args.sub_task == 'compute-scores':
      self.m_tool_chain.compute_scores(
          self.m_tool,
          groups = (self.m_args.group,),
          compute_zt_norm = False,
          indices = self.indices(self.m_file_selector.model_ids(self.m_args.group), self.m_grid.number_of_scoring_jobs),
          preload_probes = self.m_args.preload_probes,
          force = self.m_args.force)

    # concatenate
    elif self.m_args.sub_task == 'concatenate':
      self.m_tool_chain.concatenate(compute_zt_norm = False)

    # average
    elif self.m_args.sub_task == 'average-results':
      self.average_results()

    # Test if the keyword was processed
    else:
      raise ValueError("The given subtask '%s' could not be processed. THIS IS A BUG. Please report this to the authors." % self.m_args.sub_task)


  def __classification_result__(self, negatives, positives, threshold):
    return (
        bob.measure.correctly_classified_negatives(negatives, threshold).sum(dtype=numpy.float64) +
        bob.measure.correctly_classified_positives(positives, threshold).sum(dtype=numpy.float64)
      ) / float(len(positives) + len(negatives))

  def average_results(self):
    """Iterates over all the folds of the current view and computes the average result"""
    utils.info(" - Scoring: Averaging results of views %s" % self.m_args.views)
    if not self.m_args.dry_run:
      file = open(self.m_configuration.result_file, 'w')
    if 'view1' in self.m_args.views:
      if self.m_args.dry_run:
        print ("Would have averaged the results from view1 ...")
      else:
        # process the single result of view 1

        # HACK... Overwrite the score directory of the file selector to get the right result file
        self.m_file_selector.score_directories = (self.__scores_directory__('view1'),)
        res_file = self.m_file_selector.no_norm_result_file('dev')

        negatives, positives = bob.measure.load.split_four_column(res_file)
        threshold = bob.measure.eer_threshold(negatives, positives)

        far, frr = bob.measure.farfrr(negatives, positives, threshold)
        hter = (far + frr)/2.0

        file.write("On view1 (dev set only):\n\nFAR = %.3f;\tFRR = %.3f;\tHTER = %.3f;\tthreshold = %.3f\n"%(far, frr, hter, threshold))
        file.write("Classification success: %.2f%%\n\n"%(self.__classification_result__(negatives, positives, threshold) * 100.))

    if 'view2' in self.m_args.views:
      if self.m_args.dry_run:
        print ("Would have averaged the results from view2 ...")
      else:
        file.write("On view2 (eval set only):\n\n")
        # iterate over all folds of view 2
        errors = numpy.ndarray((10,), numpy.float64)
        for f in range(1,11):
          # HACK... Overwrite the score directory of the file selector to get the right result file
          self.m_file_selector.score_directories = (self.__scores_directory__('fold%d'%f),)
          dev_res_file = self.m_file_selector.no_norm_result_file('dev')
          eval_res_file = self.m_file_selector.no_norm_result_file('eval')

          # compute threshold on dev data
          dev_negatives, dev_positives = bob.measure.load.split_four_column(dev_res_file)
          threshold = bob.measure.eer_threshold(dev_negatives, dev_positives)

          # compute FAR and FRR for eval data
          eval_negatives, eval_positives = bob.measure.load.split_four_column(eval_res_file)

          far, frr = bob.measure.farfrr(eval_negatives, eval_positives, threshold)
          hter = (far + frr)/2.0

          file.write("On fold%d:\n\nFAR = %.3f;\tFRR = %.3f;\tHTER = %.3f;\tthreshold = %.3f\n"%(f, far, frr, hter, threshold))
          result = self.__classification_result__(eval_negatives, eval_positives, threshold)
          file.write("Classification success: %.2f%%\n\n"%(result * 100.))
          errors[f-1] = result

        # compute mean and std error
        mean = numpy.mean(errors)
        std = numpy.std(errors)
        file.write("\nOverall classification success: %f (with standard deviation %f)\n"%(mean,std))


def parse_args(command_line_parameters):
  """This function parses the given options (which by default are the command line options)."""
  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      conflict_handler='resolve')

  # add the arguments required for all tool chains
  config_group, dir_group, file_group, sub_dir_group, other_group, skip_group = ToolChainExecutorLFW.required_command_line_options(parser)

  # overwrite default database entry
  config_group.add_argument('-d', '--database', default = ['lfw'], nargs = '+',
      help = 'The database interface to be used. The default should work fine for the common cases.')

  sub_dir_group.add_argument('--models-directory', metavar = 'DIR', default = 'models',
      help = 'Sub-directory (of --temp-directory) where the models should be stored.')

  sub_dir_group.add_argument('--score-directory', metavar = 'DIR', default = 'nonorm',
      help = 'Sub-directory (of --user-directory) where to write the results to (used mainly to create directory structures consistent with the faceverify.py script)')

  file_group.add_argument('--result-file', '-r', type = str, metavar = 'FILE',
      help = "The file where the final results should be written into. If not specified, 'results.txt' in the --user-directory/--score-sub-directory is used.")

  skip_group.add_argument('--skip-averaging', '--noav', action='store_true',
      help = 'Skip the score averaging step.')

  #######################################################################################
  ############################ other options ############################################
  other_group.add_argument('-F', '--force', action='store_true',
      help = 'Force to erase former data if already exist.')
  other_group.add_argument('-w', '--preload-probes', action='store_true',
      help = 'Preload probe files during score computation (needs more memory, but is faster and requires fewer file accesses). WARNING! Use this flag with care!')
  other_group.add_argument('--views', nargs = '+', choices = ('view1', 'view2'), default = ['view1'],
      help = 'The views to be used, by default only the "view1" is executed.')
  other_group.add_argument('--groups', metavar = 'GROUP', nargs = '+', choices = ('dev', 'eval'), default = ['dev', 'eval'],
      help = 'The groups to compute the scores for.')

  #######################################################################################
  #################### sub-tasks being executed by this script ##########################
  parser.add_argument('--sub-task',
      choices = ('preprocess', 'train-extractor', 'extract', 'train-projector', 'project', 'train-enroller', 'enroll', 'compute-scores', 'concatenate', 'average-results'),
      help = argparse.SUPPRESS) #'Executes a subtask (FOR INTERNAL USE ONLY!!!)'
  parser.add_argument('--group', choices=('dev', 'eval'),
      help = argparse.SUPPRESS) #'The subset of the data for which the process should be executed'
  parser.add_argument('--protocol', choices = ('view1', 'fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10'),
      help = argparse.SUPPRESS) #'The protocol which should be used in this sub-task'

  #######################################################################################
  ####### shortcuts for the --skip-... commands #########################################
  skip_choices = ('preprocessing', 'extractor-training', 'extraction', 'projector-training', 'projection', 'enroller-training', 'enrollment', 'score-computation', 'concatenation', 'averaging')
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
  You just have to specify configuration scripts for any of the steps of the toolchain, which are:
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
    executor = ToolChainExecutorLFW(args, args.protocol)
    executor.execute_grid_job()
    return {}

  elif args.grid:

    # get the name of this file
    this_file = __file__
    if this_file[-1] == 'c':
      this_file = this_file[0:-1]

    # for the first protocol, we do not have any own dependencies
    dependencies = external_dependencies
    resulting_dependencies = {}
    average_dependencies = []
    dry_run_init = external_fake_job_id
    # determine which protocols should be used
    protocols=[]
    if 'view1' in args.views:
      protocols.append('view1')
    if 'view2' in args.views:
      protocols.extend(['fold%d'%i for i in range(1,11)])

    # execute all desired protocols
    for protocol in protocols:
      # create an executor object
      executor = ToolChainExecutorLFW(args, protocol)

      executor.write_info(command_line_parameters)

      executor.set_common_parameters(calling_file = this_file, parameters = command_line_parameters, fake_job_id = dry_run_init)

      # add the jobs
      new_dependencies = executor.add_jobs_to_grid(dependencies)
      resulting_dependencies.update(new_dependencies)
      if 'concatenate' in new_dependencies:
        average_dependencies.append(new_dependencies['concatenate'])
      # perform preprocessing only once for view 2
      if protocol != 'view1' and 'preprocessing' in new_dependencies:
        dependencies.append(new_dependencies['preprocessing'])
      dry_run_init += 30

    if not args.skip_averaging:
      # at the end, compute the average result
      last_dependency = executor.add_average_job_to_grid(average_dependencies)
      resulting_dependencies.update(last_dependency)

    if executor.m_grid.is_local() and args.run_local_scheduler:
      if args.dry_run:
        print ("Would have started the local scheduler to finally run the experiments with parallel jobs")
      else:
        # start the jman local deamon
        executor.execute_local_deamon()
      return {}

    # at the end of all protocols, return the list of dependencies
    return resulting_dependencies
  else:
    # not in a grid, use default tool chain sequentially

    # determine which protocols should be used
    protocols=[]
    if 'view1' in args.views:
      protocols.append('view1')
    if 'view2' in args.views:
      protocols.extend(['fold%d'%i for i in range(1,11)])

    for protocol in protocols:
      # generate executor for the current protocol
      executor = ToolChainExecutorLFW(args, protocol)
      executor.write_info(command_line_parameters)

      # execute the tool chain locally
      executor.execute_tool_chain()

    # after all protocols have been processed, compute average result
    if not args.skip_averaging:
      executor.average_results()
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
