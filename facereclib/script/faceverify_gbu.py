#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>


import sys, os
import argparse

from . import ToolChainExecutor
from .. import toolchain

class ToolChainExecutorGBU (ToolChainExecutor.ToolChainExecutor):
  
  def __init__(self, args):
    # call base class constructor
    ToolChainExecutor.ToolChainExecutor.__init__(self, args)

    # specify the file selector and tool chain objects to be used by this class (and its base class) 
    self.m_file_selector = toolchain.FileSelectorGBU(self.m_configuration)
    self.m_tool_chain = toolchain.ToolChainGBU(self.m_file_selector)
    

  def protocol_specific_configuration(self):
    """Special configuration for GBU protocol"""
    self.m_configuration.img_input_dir = self.m_database_config.img_input_dir  
    self.m_configuration.model_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.model_dir, self.m_database_config.protocol)
  
    self.m_configuration.default_extension = ".hdf5"
    
    self.m_configuration.score_dir = os.path.join(self.m_configuration.base_output_USER_dir, self.m_args.score_sub_dir, self.m_database_config.protocol) 
    
    
  def execute_tool_chain(self):
    """Executes the desired tool chain on the local machine"""
    # preprocessing
    if not self.m_args.skip_preprocessing:
      self.m_tool_chain.preprocess_images(self.m_preprocessor, force = self.m_args.force)
    # feature extraction
    if not self.m_args.skip_feature_extraction_training and hasattr(self.m_feature_extractor, 'train'):
      self.m_tool_chain.train_extractor(self.m_feature_extractor, force = self.m_args.force)
    if not self.m_args.skip_feature_extraction:
      self.m_tool_chain.extract_features(self.m_feature_extractor, force = self.m_args.force)
    # feature projection
    if not self.m_args.skip_projection_training and hasattr(self.m_tool, 'train_projector'):
      self.m_tool_chain.train_projector(self.m_tool, force = self.m_args.force)
    if not self.m_args.skip_projection and hasattr(self.m_tool, 'project'):
      self.m_tool_chain.project_features(self.m_tool, force = self.m_args.force)
    # model enrollment
    if not self.m_args.skip_enroler_training and hasattr(self.m_tool, 'train_enroler'):
      self.m_tool_chain.train_enroler(self.m_tool, force = self.m_args.force)
    if not self.m_args.skip_model_enrolment:
      self.m_tool_chain.enrol_models(self.m_tool, force = self.m_args.force)
    # score computation
    if not self.m_args.skip_score_computation:
      self.m_tool_chain.compute_scores(self.m_tool, preload_probes = self.m_args.preload_probes, force = self.m_args.force)
    self.m_tool_chain.concatenate()
    

  def add_jobs_to_grid(self, external_dependencies, perform_training = True):
    # collect job ids
    job_ids = {}
  
    # if there are any external dependencies, we need to respect them
    deps = external_dependencies[:]
    new_deps = []
    training_deps = external_dependencies[:]
    
    sets = ['training','target','query'] if perform_training else ['target','query']
    # image preprocessing; never has any dependencies.
    if not self.m_args.skip_preprocessing:
      for set in sets:
        job_ids['preprocessing_%s'%set] = self.submit_grid_job(
                '--preprocess --sub-set %s'%set, 
                name = 'pre-%s'%set, 
                list_to_split = self.m_file_selector.image_list(set), 
                number_of_files_per_job = self.m_grid_config.number_of_images_per_job, 
                dependencies = [], 
                **self.m_grid_config.preprocessing_queue)
        new_deps.append(job_ids['preprocessing_%s'%set])
      if perform_training:
        training_deps.append(job_ids['preprocessing_training'])
      deps.extend(new_deps)
      new_deps = []
      
    # feature extraction training
    if not self.m_args.skip_feature_extraction_training and perform_training and hasattr(self.m_feature_extractor, 'train'):
      job_ids['extraction_training'] = self.submit_grid_job(
              '--feature-extraction-training', 
              name = 'f-training', 
              dependencies = training_deps,
              **self.m_grid_config.training_queue)
      deps.append(job_ids['extraction_training'])
       
    if not self.m_args.skip_feature_extraction:
      for set in sets:
        job_ids['feature_extraction_%s'%set] = self.submit_grid_job(
                '--feature-extraction --sub-set %s'%set, 
                name = 'extr-%s'%set, 
                list_to_split = self.m_file_selector.preprocessed_image_list(set), 
                number_of_files_per_job = self.m_grid_config.number_of_features_per_job, 
                dependencies = deps, 
                **self.m_grid_config.extraction_queue)
        new_deps.append(job_ids['feature_extraction_%s'%set])
      if perform_training:
        training_deps.append(job_ids['feature_extraction_training'])
      deps.extend(new_deps)
      new_deps = []
      
    # feature projection training
    if not self.m_args.skip_projection_training and perform_training and hasattr(self.m_tool, 'train_projector'):
      job_ids['projector_training'] = self.submit_grid_job(
              '--train-projector', 
              name = "p-training", 
              dependencies = training_deps, 
              **self.m_grid_config.training_queue)
      deps.append(job_ids['projector_training'])
      
    if not self.m_args.skip_projection and hasattr(self.m_tool, 'project'):
      for set in sets:
        job_ids['feature_projection_%s'%set] = self.submit_grid_job(
                '--feature-projection --sub-set %s'%set, 
                list_to_split = self.m_file_selector.feature_list(set), 
                number_of_files_per_job = self.m_grid_config.number_of_projections_per_job, 
                dependencies = deps, 
                name="pro-%s"%set, 
                **self.m_grid_config.projection_queue)
        new_deps.append(job_ids['feature_projection_%s'%set])
      if perform_training:
        training_deps.append(job_ids['feature_projection_training'])
      deps.extend(new_deps)
      new_deps = []
      
    # model enrolment training
    if not self.m_args.skip_enroler_training and perform_training and hasattr(self.m_tool, 'train_enroler'):
      job_ids['enrolment_training'] = self.submit_grid_job(
              '--train-enroler', 
              dependencies = training_deps, 
              name="e-training", 
              **self.m_grid_config.training_queue)
      deps.append(job_ids['enrolment_training'])
      
    # enrol models
    if not self.m_args.skip_model_enrolment:
      job_ids['enrol'] = self.submit_grid_job(
              '--enrol-models', 
              list_to_split = self.m_file_selector.model_indices(), 
              number_of_files_per_job = self.m_grid_config.number_of_models_per_enrol_job, 
              dependencies = deps, 
              name = "enrol", 
              **self.m_grid_config.enrol_queue)
      deps.append(job_ids['enrol'])
  
    # compute scores
    if not self.m_args.skip_score_computation:
      job_ids['score'] = self.submit_grid_job(
              '--compute-scores', 
              list_to_split = self.m_file_selector.model_indices(), 
              number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job, 
              dependencies = deps, 
              name = "score", 
              **self.m_grid_config.score_queue)
      deps.append(job_ids['score'])
      
    # concatenate results   
    job_ids['concatenate'] = self.submit_grid_job(
            '--concatenate', 
            dependencies = deps, 
            name = "concat")
        
    # return the job ids, in case anyone wants to know them
    return job_ids 
  
  
  def execute_grid_job(self):
    """This function executes the grid job that is specified on the command line."""
    # preprocess
    if self.m_args.preprocess:
      self.m_tool_chain.preprocess_images(
          self.m_preprocessor, 
          sets = [self.m_args.sub_set], 
          indices = self.indices(self.m_file_selector.image_list(self.m_args.sub_set), self.m_grid_config.number_of_images_per_job), 
          force = self.m_args.force)
      
    if self.m_args.feature_extraction_training:
      self.m_tool_chain.train_extractor(
          self.m_feature_extractor, 
          force = self.m_args.force)
      
    # extract features
    if self.m_args.feature_extraction:
      self.m_tool_chain.extract_features(
          self.m_feature_extractor, 
          sets = [self.m_args.sub_set], 
          indices = self.indices(self.m_file_selector.preprocessed_image_list(self.m_args.sub_set), self.m_grid_config.number_of_features_per_job), 
          force = self.m_args.force)
      
    # train the feature projector
    if self.m_args.train_projector:
      self.m_tool_chain.train_projector(
          self.m_tool, 
          force = self.m_args.force)
      
    # project the features
    if self.m_args.projection:
      self.m_tool_chain.project_features(
          self.m_tool, 
          sets = [self.m_args.sub_set], 
          indices = self.indices(self.m_file_selector.preprocessed_image_list(self.m_args.sub_set), self.m_grid_config.number_of_projections_per_job), 
          force = self.m_args.force)
      
    # train model enroler
    if self.m_args.train_enroler:
      self.m_tool_chain.train_enroler(
          self.m_tool, 
          force = self.m_args.force)
      
    # enrol models
    if self.m_args.enrol_models:
      self.m_tool_chain.enrol_models(
          self.m_tool, 
          indices = self.indices(self.m_file_selector.model_indices(), self.m_grid_config.number_of_models_per_enrol_job), 
          force = self.m_args.force)
        
    # compute scores
    if self.m_args.compute_scores:
      self.m_tool_chain.compute_scores(
          self.m_tool, 
          indices = self.indices(self.m_file_selector.model_indices(), self.m_grid_config.number_of_models_per_score_job), 
          preload_probes = self.m_args.preload_probes, 
          force = self.m_args.force)
  
    # concatenate
    if self.m_args.concatenate:
      self.m_tool_chain.concatenate()

  
  
def parse_args(command_line_arguments = sys.argv[1:]):
  """This function parses the given options (which by default are the command line options)"""
  # sorry for that.
  global parameters
  parameters = command_line_arguments

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  # add the arguments required for all tool chains
  config_group, dir_group, file_group, sub_dir_group = ToolChainExecutorGBU.required_command_line_options(parser)

  config_group.add_argument('-D', '--database-directory', metavar = 'DIR', type = str, dest='database_dir', required = True,
      help = 'The directory containing the GBU database configuration file(s)')
  
  sub_dir_group.add_argument('--model-directory', type = str, metavar = 'DIR', dest='model_dir',default = 'models',
      help = 'Subdirectories (of temp directory) where the models should be stored')
  
  #######################################################################################
  ############################ other options ############################################
  other_group = parser.add_argument_group('\nFlags that change the behaviour of the experiment')
  other_group.add_argument('-q', '--dry-run', action='store_true', dest='dry_run',
      help = 'Only report the grid commands that will be executed, but do not execute them')
  other_group.add_argument('-f', '--force', action='store_true',
      help = 'Force to erase former data if already exist')
  other_group.add_argument('-w', '--preload-probes', action='store_true', dest='preload_probes',
      help = 'Preload probe files during score computation (needs more memory, but is faster and requires fewer file accesses). WARNING! Use this flag with care!')
  other_group.add_argument('--protocols', type = str, nargs = '+', choices = ['Good', 'Bad', 'Ugly'], default = ['Good', 'Bad', 'Ugly'],
      help = 'The protocols to use, by default all three (Good, Bad, and Ugly) are executed.')

  #######################################################################################
  ################# options for skipping parts of the toolchain #########################
  skip_group = parser.add_argument_group('\nFlags that allow to skip certain parts of the experiments. This does only make sense when the generated files are already there (e.g. when reusing parts of other experiments)')
  skip_group.add_argument('--skip-preprocessing', '--nopre', action='store_true', dest='skip_preprocessing',
      help = 'Skip the image preprocessing step')
  skip_group.add_argument('--skip-feature-extraction-training', '--nofet', action='store_true', dest='skip_feature_extraction_training',
      help = 'Skip the feature extraction training step')
  skip_group.add_argument('--skip-feature-extraction', '--nofe', action='store_true', dest='skip_feature_extraction',
      help = 'Skip the feature extraction step')
  skip_group.add_argument('--skip-projection-training', '--noprot', action='store_true', dest='skip_projection_training',
      help = 'Skip the feature extraction training')
  skip_group.add_argument('--skip-projection', '--nopro', action='store_true', dest='skip_projection',
      help = 'Skip the feature projection')
  skip_group.add_argument('--skip-enroler-training', '--noenrt', action='store_true', dest='skip_enroler_training',
      help = 'Skip the training of the model enrolment')
  skip_group.add_argument('--skip-model-enrolment', '--noenr', action='store_true', dest='skip_model_enrolment',
      help = 'Skip the model enrolment step')
  skip_group.add_argument('--skip-score-computation', '--nosc', action='store_true', dest='skip_score_computation',
      help = 'Skip the score computation step')
                      
  #######################################################################################
  #################### sub-tasks being executed by this script ##########################
  parser.add_argument('--execute-sub-task', action='store_true', dest = 'execute_sub_task',
      help = argparse.SUPPRESS) #'Executes a subtask (FOR INTERNAL USE ONLY!!!)'
  parser.add_argument('--preprocess', action='store_true', 
      help = argparse.SUPPRESS) #'Perform image preprocessing on the given range of images'
  parser.add_argument('--sub-set', type=str, choices=['training','target','query'], dest='sub_set',
      help = argparse.SUPPRESS) #'The subset of the data for which the process should be executed'
  parser.add_argument('--feature-extraction-training', action='store_true', dest = 'feature_extraction_training',
      help = argparse.SUPPRESS) #'Perform feature extraction for the given range of preprocessed images'
  parser.add_argument('--feature-extraction', action='store_true', dest = 'feature_extraction',
      help = argparse.SUPPRESS) #'Perform feature extraction for the given range of preprocessed images'
  parser.add_argument('--train-projector', action='store_true', dest = 'train_projector',
      help = argparse.SUPPRESS) #'Perform feature extraction training'
  parser.add_argument('--feature-projection', action='store_true', dest = 'projection',
      help = argparse.SUPPRESS) #'Perform feature projection'
  parser.add_argument('--train-enroler', action='store_true', dest = 'train_enroler',
      help = argparse.SUPPRESS) #'Perform enrolment training'
  parser.add_argument('--enrol-models', action='store_true', dest = 'enrol_models',
      help = argparse.SUPPRESS) #'Generate the given range of models from the features'
  parser.add_argument('--compute-scores', action='store_true', dest = 'compute_scores',
      help = argparse.SUPPRESS) #'Compute scores for the given range of models'
  parser.add_argument('--concatenate', action='store_true',
      help = argparse.SUPPRESS) #'Concatenates the results of all scores of the given group'
  
  # TODO: test which other implications of --skip-... options make sense
  
  return parser.parse_args(command_line_arguments)


def face_verify(args, external_dependencies = []):
  """This is the main entry point for computing face verification experiments.
  You just have to specify configuration scripts for any of the steps of the toolchain, which are:
  -- the database
  -- feature extraction (including image preprocessing)
  -- the score computation tool
  -- and the grid configuration (in case, the function should be executed in the grid).
  Additionally, you can skip parts of the toolchain by selecting proper --skip-... parameters.
  If your probe files are not too big, you can also specify the --preload-probes switch to speed up the score computation.
  If files should be re-generated, please specify the --force option (might be combined with the --skip-... options)"""
  
  if args.execute_sub_task:
    # execute the desired sub-task
    executor = ToolChainExecutorGBU(args)
    executor.execute_grid_job()
    return []
  
  elif args.grid:

    # get the name of this file 
    this_file = __file__
    if this_file[-1] == 'c':
      this_file = this_file[0:-1]
      
    # initialize the executor to submit the jobs to the grid 
    global parameters
    
    # for the first protocol, we do not have any own dependencies
    dependencies = external_dependencies
    resulting_dependencies = []
    perform_training = True
    dry_run_init = 0
    for protocol in args.protocols:
      # set the database
      args.database = os.path.join(args.database_dir, 'GBU_%s.py'%protocol)
      current_parameters = parameters[:]
      current_parameters.extend(['--database', args.database])
      
      # create an executor object
      executor = ToolChainExecutorGBU(args)
      executor.set_common_parameters(calling_file = this_file, parameters = current_parameters, fake_job_id = dry_run_init)

      # add the jobs
      new_dependencies = executor.add_jobs_to_grid(dependencies, perform_training = perform_training)
      resulting_dependencies.extend(new_dependencies)
      
      # select the dependencies that executes training
      if perform_training:
        for k in new_dependencies.keys():
          if "training" in k:
            dependencies.append(new_dependencies[k])
        # skip the training for the next protocol
        perform_training = False

      dry_run_init += 100
    # at the end of all protocols, return the list of dependencies
    return resulting_dependencies
  else:
    # not in a grid, use default tool chain sequentially

    for protocol in args.protocols:
      # set the database
      args.database = os.path.join(args.database_dir, 'GBU_%s.py'%protocol)
      
      executor = ToolChainExecutorGBU(args)
      # execute the tool chain locally
      executor.execute_tool_chain()
    
    # no dependencies since we executed the jobs locally
    return []
    

def main():
  """Executes the main function"""
  # do the command line parsing
  args = parse_args()
  # perform face verification test
  face_verify(args)
        
if __name__ == "__main__":
  main()  

