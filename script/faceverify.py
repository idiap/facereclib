#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import argparse
import toolchain
import os, sys, math
import imp

def config_for(args, db):
  """This function returns the configuration that is partially read from the database setup, and partially from the command line arguments"""
  # import setup of the database
  config = imp.load_source('config', args.database)
  user_name = os.environ['USER']
  if args.user_dir:
    config.base_output_USER_dir = args.user_dir
  else:
    config.base_output_USER_dir = "/idiap/user/%s/%s/%s" % (user_name, db.name, args.sub_dir)

  if args.temp_dir:
    config.base_output_TEMP_dir = args.temp_dir
  else:
    if not args.grid:
      config.base_output_TEMP_dir = "/scratch/%s/%s/%s" % (user_name, db.name, args.sub_dir)
    else:
      config.base_output_TEMP_dir = "/idiap/temp/%s/%s/%s" % (user_name, db.name, args.sub_dir)
    
  config.extractor_file = os.path.join(config.base_output_TEMP_dir, args.extractor_file)
  config.projector_file = os.path.join(config.base_output_TEMP_dir, args.projector_file) 
  config.enroler_file = os.path.join(config.base_output_TEMP_dir, args.enroler_file) 
  config.preprocessed_dir = os.path.join(config.base_output_TEMP_dir, args.preprocessed_dir) 
  config.features_dir = os.path.join(config.base_output_TEMP_dir, args.features_dir)
  config.projected_dir = os.path.join(config.base_output_TEMP_dir, args.projected_dir)
  config.models_dir = os.path.join(config.base_output_TEMP_dir, db.protocol, args.models_dirs[0])
  config.tnorm_models_dir = os.path.join(config.base_output_TEMP_dir, db.protocol, args.models_dirs[1])
  
  config.zt_norm_A_dir = os.path.join(config.base_output_TEMP_dir, db.protocol, args.score_sub_dir, args.zt_dirs[0])
  config.zt_norm_B_dir = os.path.join(config.base_output_TEMP_dir, db.protocol, args.score_sub_dir, args.zt_dirs[1])
  config.zt_norm_C_dir = os.path.join(config.base_output_TEMP_dir, db.protocol, args.score_sub_dir, args.zt_dirs[2])
  config.zt_norm_D_dir = os.path.join(config.base_output_TEMP_dir, db.protocol, args.score_sub_dir, args.zt_dirs[3])
  config.zt_norm_D_sameValue_dir = os.path.join(config.base_output_TEMP_dir, db.protocol, args.score_sub_dir, args.zt_dirs[4])
  config.default_extension = ".hdf5"
  
  config.scores_nonorm_dir = os.path.join(config.base_output_USER_dir, db.protocol, args.score_sub_dir, args.score_dirs[0]) 
  config.scores_ztnorm_dir = os.path.join(config.base_output_USER_dir, db.protocol, args.score_sub_dir, args.score_dirs[1]) 
  
  return config


def default_tool_chain(args):
  """Executes the default toolchain on the local machine"""
  db = imp.load_source('db', args.database)
  ts = imp.load_source('ts', args.tool_chain)
  pp = imp.load_source('pp', args.features)
  config = config_for(args, db)
 
  # generate File selector for the given config file
  file_selector = toolchain.FileSelector(config, db)

  # generate processing tools
  preprocessor = pp.preprocessor(pp)
  extractor = pp.feature_extractor(pp)
  tool = ts.tool(file_selector, ts)

  # generate toolchain, using the tool chain specified in the config script
  tool_chain = toolchain.ToolChain(file_selector)
  
  # compute tools without the grid
  if not args.skip_preprocessing:
    tool_chain.preprocess_images(preprocessor, force = args.force)
  if not args.skip_feature_extraction_training and hasattr(extractor, 'train'):
    tool_chain.train_extractor(extractor, force = args.force)
  if not args.skip_feature_extraction:
    tool_chain.extract_features(extractor, force = args.force)
  if not args.skip_projection_training and hasattr(tool, 'train_projector'):
    tool_chain.train_projector(tool, force = args.force)
  if not args.skip_projection and hasattr(tool, 'project'):
    tool_chain.project_features(tool, force = args.force)
  if not args.skip_enroler_training and hasattr(tool, 'train_enroler'):
    tool_chain.train_enroler(tool, force = args.force)
  if not args.skip_model_enrolment:
    tool_chain.enrol_models(tool, args.zt_norm, force = args.force)
  if not args.skip_score_computation:
    tool_chain.compute_scores(tool, args.zt_norm, preload_probes = args.preload_probes)
    if args.zt_norm:
      tool_chain.zt_norm()
    tool_chain.concatenate(args.zt_norm)
 


# Finds myself first
FACERECLIB_DIR = '/idiap/home/mguenther/Source/tools/facereclib'
# Defines the gridtk installation root - by default we look at a fixed location
# in the currently detected FACERECLIB_DIR. You can change this and hard-code
# whatever you prefer.
GRIDTK_DIR = os.path.join(FACERECLIB_DIR, 'gridtk')
sys.path.insert(0, GRIDTK_DIR)
FACERECLIB_WRAPPER = os.path.join(FACERECLIB_DIR, 'shell.py')
# The environment assures the correct execution of the wrapper
FACERECLIB_WRAPPER_ENVIRONMENT = ['FACERECLIB_DIR=%s' % FACERECLIB_DIR]
import gridtk


def generate_job_array(list_to_split, number_of_files_per_job):
  """Generates an array for the list to be split and the number of files that one job should generate""" 
  n_jobs = int(math.ceil(len(list_to_split) / float(number_of_files_per_job)))
  return (1,n_jobs,1)
  
def indices(list_to_split, number_of_files_per_job):
  """This function returns the first and last index for the files for the current job ID.
     If no job id is set (e.g., because a sub-job is executed locally), it simply returns all indices""" 
  # test if the 'SEG_TASK_ID' environment is set
  sge_task_id = os.getenv('SGE_TASK_ID')
  if sge_task_id == None:
    # task id is not set, so this function is not called from a grid job
    # hence, we process the whole list
    return (0,len(list_to_split))
  else: 
    job_id = int(sge_task_id) - 1
    # compute number of files to be executed
    start = job_id * number_of_files_per_job
    end = min((job_id + 1) * number_of_files_per_job, len(list_to_split))
    return (start, end)
  
def common_parameters():
  """This function generates a list of command line arguments that should be added in calls to sub-tasks,
     to assure that all tasks share the same setup."""
  par = ''
  for p in sys.argv[1:]:
    par += p + ' '
  return par
  

def submit(command, list_to_split, number_of_files_per_job, job_manager, common_parameters, dependencies=[], name = None, array=None, queue=None, mem=None, hostname=None, pe_opt=None):
  """Submits one job using our specialized shell wrapper. We hard-code certain
  parameters we like to use. You can change general submission parameters
  directly at this method."""
 
  cmd = [
          sys.argv[0],
          '--execute-sub-task',
          command,
          common_parameters
        ]

  if name == None:        
    name = command.split(' ')[0].replace('--','')
    log_sub_dir = command.replace(' ','__').replace('--','')
  else:
    log_sub_dir = name
  logdir = os.path.join(job_manager.temp_dir, 'logs', log_sub_dir)

  if array == None:
    array = generate_job_array(list_to_split, number_of_files_per_job)

  use_cmd = gridtk.tools.make_python_wrapper(FACERECLIB_WRAPPER, cmd)
    
  job = job_manager.submit(use_cmd, deps=dependencies, cwd=True,
      queue=queue, mem=mem, hostname=hostname, pe_opt=pe_opt,
      stdout=logdir, stderr=logdir, name=name, array=array, 
      env=FACERECLIB_WRAPPER_ENVIRONMENT)

  print 'submitted:', job
  return job


def add_grid_jobs(args):
  """This function submits calls to the grid according to the desired tool and to the given command line parameters.
     It launches only those jobs, that are necessary for the current tool, and that are not disabled with --skip... option.
     Also, it adds reasonable dependencies to the jobs so that they are executed in the right order."""
     
  # read configurations
  cfg = imp.load_source('cfg', args.grid)
  db = imp.load_source('db', args.database)
  config = config_for(args, db)
  pp = imp.load_source('pp', args.features)
  ts = imp.load_source('ts', args.tool_chain)
  
  cp = common_parameters()
 
  # generate File selector for the given config file
  file_selector = toolchain.FileSelector(config, db)
  extractor = pp.feature_extractor(pp)
  tool = ts.tool(file_selector, ts)
  # create job manager
  jm = gridtk.manager.JobManager()
  jm.temp_dir = config.base_output_TEMP_dir

  job_ids = {}

  deps = []
  # image preprocessing
  if not args.skip_preprocessing:
    job_ids['preprocessing'] = submit('--preprocess', file_selector.original_image_list(), cfg.number_of_images_per_job, jm, cp).id()
    deps.append(job_ids['preprocessing'])
    
  # feature extraction training
  if not args.skip_feature_extraction_training and hasattr(extractor, 'train'):
    job_ids['extraction_training'] = submit('--feature-extraction-training', [], 1, jm, cp, dependencies=deps, array = (1,1,1), queue='q1d', mem='8G', name='f-training').id()
    deps.append(job_ids['extraction_training'])
    

  if not args.skip_feature_extraction:
    job_ids['feature_extraction'] = submit('--feature-extraction', file_selector.preprocessed_image_list(), cfg.number_of_features_per_job, jm, cp, dependencies=deps).id()
    deps.append(job_ids['feature_extraction'])
    
  # feature projection training
  if not args.skip_projection_training and hasattr(tool, 'train_projector'):
    job_ids['training'] = submit('--train-projector', [], 1, jm, cp, dependencies=deps, array = (1,1,1), queue='q1d', mem='8G', name="p-training").id()
    deps.append(job_ids['training'])
    
  if not args.skip_projection and hasattr(tool, 'project'):
    job_ids['feature_projection'] = submit('--feature-projection', file_selector.feature_list(), cfg.number_of_projections_per_job, jm, cp, dependencies=deps).id()
    deps.append(job_ids['feature_projection'])
    
  # model enrolment training
  if not args.skip_enroler_training and hasattr(tool, 'train_enroler'):
    job_ids['enrolment_training'] = submit('--train-enroler', [], 1, jm, cp, dependencies=deps, array = (1,1,1), queue='q1d', mem='8G', name="e-training").id()
    deps.append(job_ids['enrolment_training'])
    
    
  # enrol models
  enrol_deps_N = {}
  enrol_deps_T = {}
  score_deps = {}
  concat_deps = {}
  for group in ['dev', 'eval']:
    enrol_deps_N[group] = deps[:]
    enrol_deps_T[group] = deps[:]
    if not args.skip_model_enrolment:
      job_ids['model_%s_N'%group] = submit('--enrol-models --group=%s --model-type=N'%group, file_selector.model_ids(group), cfg.number_of_models_per_enrol_job, jm, cp, dependencies=deps, name = "enrol-N-%s"%group).id()
      enrol_deps_N[group].append(job_ids['model_%s_N'%group])

      if args.zt_norm:
        job_ids['model_%s_T'%group] = submit('--enrol-models --group=%s --model-type=T'%group, file_selector.Tmodel_ids(group), cfg.number_of_models_per_enrol_job, jm, cp, dependencies=deps, name = "enrol-T-%s"%group).id()
        enrol_deps_T[group].append(job_ids['model_%s_T'%group])
        
    # compute A,B,C, and D scores
    if not args.skip_score_computation:
      job_ids['score_%s_A'%group] = submit('--compute-scores --group=%s --score-type=A'%group, file_selector.model_ids(group), cfg.number_of_models_per_score_job, jm, cp, dependencies=enrol_deps_N[group], name = "score-A-%s"%group).id()
      concat_deps[group] = [job_ids['score_%s_A'%group]]
      if args.zt_norm:
        job_ids['score_%s_B'%group] = submit('--compute-scores --group=%s --score-type=B'%group, file_selector.model_ids(group), cfg.number_of_models_per_score_job, jm, cp, dependencies=enrol_deps_N[group], name = "score-B-%s"%group).id()
        job_ids['score_%s_C'%group] = submit('--compute-scores --group=%s --score-type=C'%group, file_selector.Tmodel_ids(group), cfg.number_of_models_per_score_job, jm, cp, dependencies=enrol_deps_T[group], name = "score-C-%s"%group).id()
        job_ids['score_%s_D'%group] = submit('--compute-scores --group=%s --score-type=D'%group, file_selector.Tmodel_ids(group), cfg.number_of_models_per_score_job, jm, cp, dependencies=enrol_deps_T[group], name = "score-D-%s"%group).id()
        
        # compute zt-norm
        score_deps[group] = [job_ids['score_%s_A'%group], job_ids['score_%s_B'%group], job_ids['score_%s_C'%group], job_ids['score_%s_D'%group]]
        job_ids['score_%s_Z'%group] = submit('--compute-scores --group=%s --score-type=Z'%group, [], 1, jm, cp, dependencies=score_deps[group], array = (1,1,1), name = "score-Z-%s"%group).id()
        concat_deps[group].extend([job_ids['score_%s_B'%group], job_ids['score_%s_C'%group], job_ids['score_%s_D'%group], job_ids['score_%s_Z'%group]])
        
      # concatenate results   
      submit('--concatenate --group=%s'%group, [], 1, jm, cp, dependencies=concat_deps[group], array = (1,1,1), name = "concat-%s"%group)
    

def execute_grid_job(args):
  """This function executes the grid job that is specified on the command line."""
  # read configurations
  cfg = imp.load_source('cfg', args.grid)
  db = imp.load_source('db', args.database)
  ts = imp.load_source('ts', args.tool_chain)
  pp = imp.load_source('pp', args.features)
  config = config_for(args, db)
  
  # generate File selector for the given config file
  file_selector = toolchain.FileSelector(config, db)
  # generate image preprocessor
  preprocessor = pp.preprocessor(pp)
  extractor = pp.feature_extractor(pp)
  tool = ts.tool(file_selector, ts)
  # generate toolchain, using the tool chain specified in the config script
  tool_chain = toolchain.ToolChain(file_selector)
  
  # preprocess
  if args.preprocess:
    tool_chain.preprocess_images(preprocessor, indices = indices(file_selector.original_image_list(), cfg.number_of_images_per_job), force = args.force)
    
  if args.feature_extraction_training:
    tool_chain.train_extractor(extractor, force = args.force)
    
  # extract features
  if args.feature_extraction:
    tool_chain.extract_features(extractor, indices = indices(file_selector.preprocessed_image_list(), cfg.number_of_features_per_job), force = args.force)
    
  # train the feature projector
  if args.train_projector:
    tool_chain.train_projector(tool, force = args.force)
    
  # project the features
  if args.projection:
    tool_chain.project_features(tool, indices = indices(file_selector.preprocessed_image_list(), cfg.number_of_projections_per_job), force = args.force)
    
  # train model enroler
  if args.train_enroler:
    tool_chain.train_enroler(tool, force = args.force)
    
  # enrol models
  if args.enrol_models:
    if args.model_type == 'N':
      tool_chain.enrol_models(tool, args.zt_norm, indices = indices(file_selector.model_ids(args.group), cfg.number_of_models_per_enrol_job), groups=[args.group], types = ['N'], force = args.force)
    else:
      tool_chain.enrol_models(tool, args.zt_norm, indices = indices(file_selector.Tmodel_ids(args.group), cfg.number_of_models_per_enrol_job), groups=[args.group], types = ['T'], force = args.force)
      
  # compute scores
  if args.compute_scores:
    if args.score_type in ['A', 'B']:
      tool_chain.compute_scores(tool, args.zt_norm, indices = indices(file_selector.model_ids(args.group), cfg.number_of_models_per_score_job), groups=[args.group], types = [args.score_type], preload_probes = args.preload_probes)
    elif args.score_type in ['C', 'D']:
      tool_chain.compute_scores(tool, args.zt_norm, indices = indices(file_selector.Tmodel_ids(args.group), cfg.number_of_models_per_score_job), groups=[args.group], types = [args.score_type], preload_probes = args.preload_probes)
    else:
      tool_chain.zt_norm(groups=[args.group])
  # concatenate
  if args.concatenate:
    tool_chain.concatenate(args.zt_norm, groups=[args.group])
  

def main():
  """This is the main entry point for computing face verification experiments.
  You just have to specify configuration scripts for any of the steps of the toolchain, which are:
  -- the database
  -- feature extraction (including image preprocessing)
  -- the score computation tool
  -- and the grid configuration (in case, the function should be executed in the grid).
  Additionally, you can skip parts of the toolchain by selecting proper --skip-... parameters.
  If your probe files are not too big, you can also specify the --preload-probes switch to speed up the score computation.
  If files should be re-generated, please specify the --force option (might be combined with the --skip-... options)"""
  
  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  #######################################################################################
  ############## options that are required to be specified #######################
  parser.add_argument('-d', '--database', metavar = 'FILE', type = str, required = True,
                      help = 'The database configuration file')
  parser.add_argument('-t', '--tool-chain', type = str, dest = 'tool_chain', required = True, metavar = 'FILE', 
                      help = 'The tool chain configuration file')
  parser.add_argument('-p', '--features-extraction', metavar = 'FILE', type = str, dest='features', 
                      help = 'Configuration script for preprocessing the images and extracting the features')
  
  parser.add_argument('-g', '--grid', metavar = 'FILE', type = str, 
                      help = 'Configuration file for the grid setup; if not specified, the commands are executed on the local machine')

  #######################################################################################
  ############## options to modify default directories or file names ####################
  parser.add_argument('-T', '--temp-dir', metavar = 'DIR', type = str, dest = 'temp_dir',
                      help = 'The directory for temporary files; defaults to /idiap/temp/$USER/database-name/sub-dir (or /scratch/$USER/database-name/sub-dir, when executed locally)')
  parser.add_argument('-U', '--user-dir', metavar = 'DIR', type = str, dest = 'user_dir',
                      help = 'The directory for temporary files; defaults to /idiap/user/$USER/database-name/sub-dir')
  parser.add_argument('-b', '--sub-dir', type = str, dest = 'sub_dir', default = 'default',
                      help = 'The sub-directory where the results of the current experiment should be stored.')
  parser.add_argument('-s', '--score-sub-dir', type = str, dest = 'score_sub_dir', default = 'scores',
                      help = 'The sub-directory where to write the scores to.')
  
  parser.add_argument('--extractor-file', type = str, metavar = 'FILE', default = 'Extractor.hdf5',
                      help = 'Name of the file to write the feature extractor into')
  parser.add_argument('--projector-file', type = str, metavar = 'FILE', default = 'Projector.hdf5',
                      help = 'Name of the file to write the feature projector into')
  parser.add_argument('--enroler-file' , type = str, metavar = 'FILE', default = 'Enroler.hdf5',
                      help = 'Name of the file to write the model enroler into')
  parser.add_argument('--preprocessed-image-directory', type = str, metavar = 'DIR', default = 'preprocessed', dest = 'preprocessed_dir',
                      help = 'Name of the directory of the preprocessed images')
  parser.add_argument('--features-directory', type = str, metavar = 'DIR', default = 'features', dest = 'features_dir',
                      help = 'Name of the directory of the features')
  parser.add_argument('--projected-directory', type = str, metavar = 'DIR', default = 'projected', dest = 'projected_dir',
                      help = 'Name of the directory where the projected data should be stored')
  parser.add_argument('--models-directories', type=str, metavar = 'DIR', nargs = 2, dest='models_dirs',
                      default = ['models', 'tmodels'],
                      help = 'Subdirectories (of temp directory) where the models should be stored')
  parser.add_argument('--zt-norm-directories', type = str, metavar = 'DIR', nargs = 5, dest='zt_dirs', 
                      default = ['zt_norm_A', 'zt_norm_B', 'zt_norm_C', 'zt_norm_D', 'zt_norm_D_sameValue'],
                      help = 'Subdiretories (of temp directory) where to write the zt_norm values')
  parser.add_argument('--score-dirs', type = str, metavar = 'DIR', nargs = 2, dest='score_dirs',
                      default = ['nonorm', 'ztnorm'],
                      help = 'Subdirectories (of user directories) where to write the results to')
  
  #######################################################################################
  ############################ other options ############################################
  parser.add_argument('-z', '--zt-norm', action='store_false', dest = 'zt_norm',
                      help = 'DISABLE the computation of ZT norms')
  parser.add_argument('-f', '--force', action='store_true',
                      help = 'Force to erase former data if already exist')
  parser.add_argument('-w', '--preload-probes', action='store_true', dest='preload_probes',
                      help = 'Preload probe files during score computation (needs more memory, but is faster and requires fewer file accesses). WARNING! Use this flag with care!')

  #######################################################################################
  ################# options for skipping parts of the toolchain #########################
  parser.add_argument('--skip-preprocessing', '--nopre', action='store_true', dest='skip_preprocessing',
                      help = 'Skip the image preprocessing step')
  parser.add_argument('--skip-feature-extraction-training', '--nofet', action='store_true', dest='skip_feature_extraction_training',
                      help = 'Skip the feature extraction training step')
  parser.add_argument('--skip-feature-extraction', '--nofe', action='store_true', dest='skip_feature_extraction',
                      help = 'Skip the feature extraction step')
  parser.add_argument('--skip-projection-training', '--noprot', action='store_true', dest='skip_projection_training',
                      help = 'Skip the feature extraction training')
  parser.add_argument('--skip-projection', '--nopro', action='store_true', dest='skip_projection',
                      help = 'Skip the feature projection')
  parser.add_argument('--skip-enroler-training', '--noenrt', action='store_true', dest='skip_enroler_training',
                      help = 'Skip the training of the model enrolment')
  parser.add_argument('--skip-model-enrolment', '--noenr', action='store_true', dest='skip_model_enrolment',
                      help = 'Skip the model enrolment step')
  parser.add_argument('--skip-score-computation', '--nosc', action='store_true', dest='skip_score_computation',
                      help = 'Skip the score computation step')
                      
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
  
  args = parser.parse_args()
  
  # TODO: test which other implications of --skip-... options make sense
  
  # as the main entry point, check whether the grid option was given
  if not args.grid:
    # not in a grid, use default tool chain sequentially
    default_tool_chain(args)
    
  elif args.execute_sub_task:
    # execute the desired sub-task
    execute_grid_job(args)
  else:
    # no other parameter given, so deploy new jobs
    add_grid_jobs(args)
    
        
if __name__ == "__main__":
  main()
