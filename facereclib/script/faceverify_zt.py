#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>


import sys, os
import argparse

from . import ToolChainExecutor
from .. import toolchain

class ToolChainExecutorZT (ToolChainExecutor.ToolChainExecutor):
  """Class that executes the ZT tool chain (locally or in the grid)"""
  
  def __init__(self, args):
    # call base class constructor
    ToolChainExecutor.ToolChainExecutor.__init__(self, args)

    # specify the file selector and tool chain objects to be used by this class (and its base class) 
    self.m_file_selector = toolchain.FileSelectorZT(self.m_configuration, self.m_database_config)
    self.m_tool_chain = toolchain.ToolChainZT(self.m_file_selector)
    
    
  def protocol_specific_configuration(self):
    """Special configuration for ZT-Norm computation"""
    self.m_configuration.models_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.models_dirs[0], self.m_database_config.protocol)
    self.m_configuration.tnorm_models_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.models_dirs[1], self.m_database_config.protocol)
    
    self.m_configuration.zt_norm_A_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_database_config.protocol, self.m_args.zt_dirs[0])
    self.m_configuration.zt_norm_B_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_database_config.protocol, self.m_args.zt_dirs[1])
    self.m_configuration.zt_norm_C_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_database_config.protocol, self.m_args.zt_dirs[2])
    self.m_configuration.zt_norm_D_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_database_config.protocol, self.m_args.zt_dirs[3])
    self.m_configuration.zt_norm_D_sameValue_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_database_config.protocol, self.m_args.zt_dirs[4])
    self.m_configuration.default_extension = ".hdf5"
    
    self.m_configuration.scores_nonorm_dir = os.path.join(self.m_configuration.base_output_USER_dir, self.m_args.score_sub_dir, self.m_database_config.protocol, self.m_args.score_dirs[0]) 
    self.m_configuration.scores_ztnorm_dir = os.path.join(self.m_configuration.base_output_USER_dir, self.m_args.score_sub_dir, self.m_database_config.protocol, self.m_args.score_dirs[1]) 
    
    
  def execute_tool_chain(self):
    """Executes the ZT tool chain on the local machine"""
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
      self.m_tool_chain.project_features(self.m_tool, force = self.m_args.force, extractor = self.m_feature_extractor)
    # model enrolment
    if not self.m_args.skip_enroler_training and hasattr(self.m_tool, 'train_enroler'):
      self.m_tool_chain.train_enroler(self.m_tool, force = self.m_args.force)
    if not self.m_args.skip_model_enrolment:
      self.m_tool_chain.enrol_models(self.m_tool, self.m_feature_extractor, not self.m_args.no_zt_norm, groups = self.m_args.groups, force = self.m_args.force)
    # score computation
    if not self.m_args.skip_score_computation:
      self.m_tool_chain.compute_scores(self.m_tool, not self.m_args.no_zt_norm, groups = self.m_args.groups, preload_probes = self.m_args.preload_probes, force = self.m_args.force)
      if not self.m_args.no_zt_norm:
        self.m_tool_chain.zt_norm(groups = self.m_args.groups)
    self.m_tool_chain.concatenate(not self.m_args.no_zt_norm, groups = self.m_args.groups)
  
    
  def add_jobs_to_grid(self, external_dependencies):
    """Adds all (desired) jobs of the tool chain to the grid"""
    # collect the job ids
    job_ids = {}
  
    # if there are any external dependencies, we need to respect them
    deps = external_dependencies[:]
    
    # image preprocessing; never has any dependencies.
    if not self.m_args.skip_preprocessing:
      job_ids['preprocessing'] = self.submit_grid_job(
              '--preprocess', 
              list_to_split = self.m_file_selector.original_image_list(), 
              number_of_files_per_job = self.m_grid_config.number_of_images_per_job, 
              dependencies = [], 
              **self.m_grid_config.preprocessing_queue)
      deps.append(job_ids['preprocessing'])
      
    # feature extraction training
    if not self.m_args.skip_feature_extraction_training and hasattr(self.m_feature_extractor, 'train'):
      job_ids['extraction_training'] = self.submit_grid_job(
              '--feature-extraction-training', 
              name = 'f-training',  
              dependencies = deps, 
              **self.m_grid_config.training_queue)
      deps.append(job_ids['extraction_training'])
      
    # feature extraction
    if not self.m_args.skip_feature_extraction:
      job_ids['feature_extraction'] = self.submit_grid_job(
              '--feature-extraction', 
              list_to_split = self.m_file_selector.preprocessed_image_list(), 
              number_of_files_per_job = self.m_grid_config.number_of_features_per_job, 
              dependencies = deps, 
              **self.m_grid_config.extraction_queue)
      deps.append(job_ids['feature_extraction'])
      
    # feature projection training
    if not self.m_args.skip_projection_training and hasattr(self.m_tool, 'train_projector'):
      job_ids['projector_training'] = self.submit_grid_job(
              '--train-projector', 
              name="p-training", 
              dependencies = deps, 
              **self.m_grid_config.training_queue)
      deps.append(job_ids['projector_training'])
    
    # feature projection
    if not self.m_args.skip_projection and hasattr(self.m_tool, 'project'):
      job_ids['feature_projection'] = self.submit_grid_job(
              '--feature-projection', 
              list_to_split = self.m_file_selector.feature_list(),
              number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
              dependencies = deps, 
              **self.m_grid_config.projection_queue)
      deps.append(job_ids['feature_projection'])
      
    # model enrolment training
    if not self.m_args.skip_enroler_training and hasattr(self.m_tool, 'train_enroler'):
      job_ids['enrolment_training'] = self.submit_grid_job(
              '--train-enroler', 
              name = "e-training",
              dependencies = deps, 
              **self.m_grid_config.training_queue)
      deps.append(job_ids['enrolment_training'])
       
    # enrol models
    enrol_deps_n = {}
    enrol_deps_t = {}
    score_deps = {}
    concat_deps = {}
    for group in self.m_args.groups:
      enrol_deps_n[group] = deps[:]
      enrol_deps_t[group] = deps[:]
      if not self.m_args.skip_model_enrolment:
        job_ids['enrol_%s_N'%group] = self.submit_grid_job(
                '--enrol-models --group=%s --model-type=N'%group, 
                name = "enrol-N-%s"%group,  
                list_to_split = self.m_file_selector.model_ids(group), 
                number_of_files_per_job = self.m_grid_config.number_of_models_per_enrol_job, 
                dependencies = deps, 
                **self.m_grid_config.enrol_queue)
        enrol_deps_n[group].append(job_ids['enrol_%s_N'%group])
  
        if not self.m_args.no_zt_norm:
          job_ids['enrol_%s_T'%group] = self.submit_grid_job(
                  '--enrol-models --group=%s --model-type=T'%group,
                  name = "enrol-T-%s"%group, 
                  list_to_split = self.m_file_selector.tmodel_ids(group), 
                  number_of_files_per_job = self.m_grid_config.number_of_models_per_enrol_job, 
                  dependencies = deps,
                  **self.m_grid_config.enrol_queue)
          enrol_deps_t[group].append(job_ids['enrol_%s_T'%group])
          
      # compute A,B,C, and D scores
      if not self.m_args.skip_score_computation:
        job_ids['score_%s_A'%group] = self.submit_grid_job(
                '--compute-scores --group=%s --score-type=A'%group, 
                name = "score-A-%s"%group, 
                list_to_split = self.m_file_selector.model_ids(group), 
                number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job, 
                dependencies = enrol_deps_n[group], 
                **self.m_grid_config.score_queue)
        concat_deps[group] = [job_ids['score_%s_A'%group]]
        
        if not self.m_args.no_zt_norm:
          job_ids['score_%s_B'%group] = self.submit_grid_job(
                  '--compute-scores --group=%s --score-type=B'%group, 
                  name = "score-B-%s"%group, 
                  list_to_split = self.m_file_selector.model_ids(group), 
                  number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job, 
                  dependencies = enrol_deps_n[group], 
                  **self.m_grid_config.score_queue)
          
          job_ids['score_%s_C'%group] = self.submit_grid_job(
                  '--compute-scores --group=%s --score-type=C'%group, 
                  name = "score-C-%s"%group, 
                  list_to_split = self.m_file_selector.tmodel_ids(group), 
                  number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job, 
                  dependencies = enrol_deps_t[group], 
                  **self.m_grid_config.score_queue)
                  
          job_ids['score_%s_D'%group] = self.submit_grid_job(
                  '--compute-scores --group=%s --score-type=D'%group, 
                  name = "score-D-%s"%group, 
                  list_to_split = self.m_file_selector.tmodel_ids(group), 
                  number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job, 
                  dependencies = enrol_deps_t[group], 
                  **self.m_grid_config.score_queue)
          
          # compute zt-norm
          score_deps[group] = [job_ids['score_%s_A'%group], job_ids['score_%s_B'%group], job_ids['score_%s_C'%group], job_ids['score_%s_D'%group]]
          job_ids['score_%s_Z'%group] = self.submit_grid_job(
                  '--compute-scores --group=%s --score-type=Z'%group,
                  name = "score-Z-%s"%group,
                  dependencies = score_deps[group]) 
          concat_deps[group].extend([job_ids['score_%s_B'%group], job_ids['score_%s_C'%group], job_ids['score_%s_D'%group], job_ids['score_%s_Z'%group]])
          
      # concatenate results   
      job_ids['concat_%s'%group] = self.submit_grid_job(
              '--concatenate --group=%s'%group,
              name = "concat-%s"%group,
              dependencies = concat_deps[group])
        
    # return the job ids, in case anyone wants to know them
    return job_ids 
  

  def execute_grid_job(self):
    """Run the desired job of the ZT tool chain that is specified on command line""" 
    # preprocess
    if self.m_args.preprocess:
      self.m_tool_chain.preprocess_images(
          self.m_preprocessor, 
          indices = self.indices(self.m_file_selector.original_image_list(), self.m_grid_config.number_of_images_per_job), 
          force = self.m_args.force)
      
    if self.m_args.feature_extraction_training:
      self.m_tool_chain.train_extractor(
          self.m_feature_extractor, 
          force = self.m_args.force)
      
    # extract features
    if self.m_args.feature_extraction:
      self.m_tool_chain.extract_features(
          self.m_feature_extractor, 
          indices = self.indices(self.m_file_selector.preprocessed_image_list(), self.m_grid_config.number_of_features_per_job), 
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
          indices = self.indices(self.m_file_selector.preprocessed_image_list(), self.m_grid_config.number_of_projections_per_job), 
          force = self.m_args.force,
          extractor = self.m_feature_extractor)
      
    # train model enroler
    if self.m_args.train_enroler:
      self.m_tool_chain.train_enroler(
          self.m_tool, 
          force = self.m_args.force)
      
    # enrol models
    if self.m_args.enrol_models:
      if self.m_args.model_type == 'N':
        self.m_tool_chain.enrol_models(
            self.m_tool,
            self.m_feature_extractor,
            not self.m_args.no_zt_norm, 
            indices = self.indices(self.m_file_selector.model_ids(self.m_args.group), self.m_grid_config.number_of_models_per_enrol_job), 
            groups = [self.m_args.group], 
            types = ['N'], 
            force = self.m_args.force)

      else:
        self.m_tool_chain.enrol_models(
            self.m_tool,
            self.m_feature_extractor,
            not self.m_args.no_zt_norm, 
            indices = self.indices(self.m_file_selector.tmodel_ids(self.m_args.group), self.m_grid_config.number_of_models_per_enrol_job), 
            groups = [self.m_args.group], 
            types = ['T'], 
            force = self.m_args.force)
        
    # compute scores
    if self.m_args.compute_scores:
      if self.m_args.score_type in ['A', 'B']:
        self.m_tool_chain.compute_scores(
            self.m_tool, 
            not self.m_args.no_zt_norm, 
            indices = self.indices(self.m_file_selector.model_ids(self.m_args.group), self.m_grid_config.number_of_models_per_score_job), 
            groups = [self.m_args.group], 
            types = [self.m_args.score_type], 
            preload_probes = self.m_args.preload_probes, 
            force = self.m_args.force)

      elif self.m_args.score_type in ['C', 'D']:
        self.m_tool_chain.compute_scores(
            self.m_tool, 
            not self.m_args.no_zt_norm, 
            indices = self.indices(self.m_file_selector.tmodel_ids(self.m_args.group), self.m_grid_config.number_of_models_per_score_job), 
            groups = [self.m_args.group], 
            types = [self.m_args.score_type], 
            preload_probes = self.m_args.preload_probes, 
            force = self.m_args.force)

      else:
        self.m_tool_chain.zt_norm(groups = [self.m_args.group])
    # concatenate
    if self.m_args.concatenate:
      self.m_tool_chain.concatenate(
          not self.m_args.no_zt_norm, 
          groups = [self.m_args.group])



def parse_args(command_line_arguments = sys.argv[1:]):
  """This function parses the given options (which by default are the command line options)"""
  # sorry for that.
  global parameters
  parameters = command_line_arguments

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # add the arguments required for all tool chains
  config_group, dir_group, file_group, sub_dir_group, other_group, skip_group = ToolChainExecutorZT.required_command_line_options(parser)
  
  sub_dir_group.add_argument('--models-directories', type = str, metavar = 'DIR', nargs = 2, dest='models_dirs',
      default = ['models', 'tmodels'],
      help = 'Subdirectories (of temp directory) where the models should be stored')
  sub_dir_group.add_argument('--zt-norm-directories', type = str, metavar = 'DIR', nargs = 5, dest='zt_dirs', 
      default = ['zt_norm_A', 'zt_norm_B', 'zt_norm_C', 'zt_norm_D', 'zt_norm_D_sameValue'],
      help = 'Subdirectories (of --temp-dir) where to write the zt_norm values')
  sub_dir_group.add_argument('--score-dirs', type = str, metavar = 'DIR', nargs = 2, dest='score_dirs',
      default = ['nonorm', 'ztnorm'],
      help = 'Subdirectories (of --user-dir) where to write the results to')
  
  #######################################################################################
  ############################ other options ############################################
  other_group.add_argument('-z', '--no-zt-norm', action='store_true', dest = 'no_zt_norm',
      help = 'DISABLE the computation of ZT norms')
  other_group.add_argument('-F', '--force', action='store_true',
      help = 'Force to erase former data if already exist')
  other_group.add_argument('-w', '--preload-probes', action='store_true', dest='preload_probes',
      help = 'Preload probe files during score computation (needs more memory, but is faster and requires fewer file accesses). WARNING! Use this flag with care!')
  other_group.add_argument('--groups', type = str,  metavar = 'GROUP', nargs = '+', default = ['dev', 'eval'],
      help = "The group (i.e., 'dev' or  'eval') for which the models and scores should be generated")

  #######################################################################################
  #################### sub-tasks being executed by this script ##########################
  parser.add_argument('--execute-sub-task', action='store_true', dest = 'execute_sub_task',
      help = argparse.SUPPRESS) #'Executes a subtask (FOR INTERNAL USE ONLY!!!)'
  parser.add_argument('--preprocess', action='store_true', 
      help = argparse.SUPPRESS) #'Perform image preprocessing on the given range of images'
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
  parser.add_argument('--model-type', type = str, choices = ['N', 'T'], metavar = 'TYPE', 
      help = argparse.SUPPRESS) #'Which type of models to generate (Normal or TModels)'
  parser.add_argument('--compute-scores', action='store_true', dest = 'compute_scores',
      help = argparse.SUPPRESS) #'Compute scores for the given range of models'
  parser.add_argument('--score-type', type = str, choices=['A', 'B', 'C', 'D', 'Z'],  metavar = 'SCORE', 
      help = argparse.SUPPRESS) #'The type of scores that should be computed'
  parser.add_argument('--group', type = str,  metavar = 'GROUP', 
      help = argparse.SUPPRESS) #'The group for which the current action should be performed'
  parser.add_argument('--concatenate', action='store_true',
      help = argparse.SUPPRESS) #'Concatenates the results of all scores of the given group'
  
  return parser.parse_args(command_line_arguments)


def face_verify(args, external_dependencies = [], external_fake_job_id = 0):
  """This is the main entry point for computing face verification experiments.
  You just have to specify configuration scripts for any of the steps of the toolchain, which are:
  -- the database
  -- feature extraction (including image preprocessing)
  -- the score computation tool
  -- and the grid configuration (in case, the function should be executed in the grid).
  Additionally, you can skip parts of the toolchain by selecting proper --skip-... parameters.
  If your probe files are not too big, you can also specify the --preload-probes switch to speed up the score computation.
  If files should be re-generated, please specify the --force option (might be combined with the --skip-... options)"""
  
  
  # generate tool chain executor
  executor = ToolChainExecutorZT(args)
  # as the main entry point, check whether the grid option was given
  if not args.grid:
    # not in a grid, use default tool chain sequentially
    executor.execute_tool_chain()
    return []
    
  elif args.execute_sub_task:
    # execute the desired sub-task
    executor.execute_grid_job()
    return []
  else:
    # no other parameter given, so deploy new jobs

    # get the name of this file 
    this_file = __file__
    if this_file[-1] == 'c':
      this_file = this_file[0:-1]
      
    # initialize the executor to submit the jobs to the grid 
    global parameters
    executor.set_common_parameters(calling_file = this_file, parameters = parameters, fake_job_id = external_fake_job_id )
    
    # add the jobs
    return executor.add_jobs_to_grid(external_dependencies)
    

def main():
  """Executes the main function"""
  # do the command line parsing
  args = parse_args()
  # perform face verification test
  face_verify(args)
        
if __name__ == "__main__":
  main()  

