#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import argparse
import toolchain
import os
import imp


def main():
  """This is the main entry point for computing face recognition experiments.
  You just have to specify the configuration script, and everything els will be computed automatically."""
  
  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-c', '--config-file', metavar='FILE', type=str, required = True,
      dest='config_file', default="", help='Filename of the configuration file to use to run the script on the grid (defaults to "%(default)s")')
  parser.add_argument('-f', '--force', dest='force', action='store_true',
      default=False, help='Force to erase former data if already exist')
  args = parser.parse_args()

  # read config file
  config = imp.load_source('config', args.config_file)
  # generate File selector for the given config file
  file_selector = toolchain.FileSelector.FileSelector(config)

  # generate toolchain, using the tool chain specified in the config script
  tool_chain = config.tool_chain(file_selector)
      
  # perform the tool chain
  tool_chain.preprocess_images(config, args.force)
  tool_chain.train_system(args.force)
  tool_chain.extract_features(args.force)
  tool_chain.enrol_models(config.zt_norm, args.force)
  tool_chain.compute_scores(config.zt_norm)
  if config.zt_norm:
    tool_chain.zt_norm()
  tool_chain.concatenate(config.zt_norm)
  
  # That's it. We are done. Please check your results. :)

def config_for(args, db):
  # import setup of the database
  config = imp.load_source('config', args.database)
  user_name = os.environ['USER']
  if args.user_dir:
    config.base_output_USER_dir = args.user_dir
  else:
    config.base_output_USER_dir = "/idiap/user/%s/%s" % (database.name, args.sub_dir)

  if args.temp_dir:
    config.base_output_USER_dir = args.temp_dir
  else:
    config.base_output_USER_dir = "/idiap/temp/%s/%s" % (database.name, args.sub_dir)
    
  config.trained_model_file = os.path.join(config.base_output_TEMP_dir, args.trained_model_file) 
  config.preprocessed_dir = os.path.join(config.base_output_TEMP_dir, args.preprocessed_dir) 
  config.features_dir = os.path.join(config.base_output_TEMP_dir, args.features_dir)
  config.models_dir = os.path.join(config.base_output_TEMP_dir, args.models_dirs[0])
  config.tnorm_models_dir = os.path.join(config.base_output_TEMP_dir, args.models_dirs[1])
  
  config.zt_norm_A_dir = os.path.join(config.base_output_TEMP_dir, db.protocol, args.zt_dirs[0])
  config.zt_norm_B_dir = os.path.join(config.base_output_TEMP_dir, db.protocol, args.zt_dirs[1])
  config.zt_norm_C_dir = os.path.join(config.base_output_TEMP_dir, db.protocol, args.zt_dirs[2])
  config.zt_norm_D_dir = os.path.join(config.base_output_TEMP_dir, db.protocol, args.zt_dirs[3])
  config.zt_norm_D_sameValue_dir = os.path.join(config.base_output_TEMP_dir, db.protocol, args.zt_dirs[4])
  config.default_extension = ".hdf5"
  
  config.scores_nonorm_dir = os.path.join(config.base_output_USER_dir, db.protocol, args.score_dirs[0]) 
  config.scores_ztnorm_dir = os.path.join(config.base_output_USER_dir, db.protocol, args.score_dirs[1]) 
  
  return config


def default_chain(args):
  db = imp.load_source('db', args.database)
  ts = imp.load_source('ts', args.tool_chain)
  config = config_for(args, db)
 
  # generate File selector for the given config file
  file_selector = toolchain.FileSelector.FileSelector(config, db)

  # generate toolchain, using the tool chain specified in the config script
  tool_chain = ts.tool_chain(file_selector, ts)

  # compute tools without the grid
  tool_chain.preprocess_images(config, args.force)
  tool_chain.train_system(args.force)
  tool_chain.extract_features(args.force)
  tool_chain.enrol_models(config.zt_norm, args.force)
  tool_chain.compute_scores(config.zt_norm)
  if config.zt_norm:
    tool_chain.zt_norm()
  tool_chain.concatenate(config.zt_norm)
 

def main2():
  """This is the main entry point for computing face recognition experiments.
  You just have to specify configuration scripts for any of the steps of the toolchain."""
  
  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-T', '--temp-dir', metavar = 'DIR', type = str, dest = 'temp_dir',
                      help = 'The directory for temporary files; defaults to /idiap/temp/$USER/database-name/sub-dir')
  parser.add_argument('-U', '--user-dir', metavar = 'DIR', type = str, dest = 'user_dir',
                      help = 'The directory for temporary files; defaults to /idiap/user/$USER/database-name/sub-dir')
  parser.add_argument('-b', '--sub-dir', type = str, dest = 'sub_dir', default = 'default',
                      help = 'The sub-directory where the results of the current experiment should be stored')
  
  parser.add_argument('-d', '--database', metavar = 'FILE', type = str, required = True,
                      help = 'The database configuration file')
  parser.add_argument('-t', '--tool-chain', type = str, dest = 'tool_chain', required = True,
                      help = 'The tool chain configuration file')
  parser.add_argument('-p', '--preprecessing-config', metavar = 'FILE', type = str, 
                      help = 'Configuration script for preprocessing the images')
  parser.add_argument('--trained-model-file', type = str, default = 'Extractor.hdf5',
                      help = 'Name of the file to write the extraction model into')
  parser.add_argument('--preprocessed-image-directory', type = str, default = 'preprocessed', dest = 'preprocessed_dir',
                      help = 'Name of the directory of the preprocessed images')
  parser.add_argument('--feature-directory', type = str, default = 'features', dest = 'feature_dir',
                      help = 'Name of the directory of the features')
  parser.add_argument('--models-directories', type=str, nargs = 2, dest='models_dirs',
                      default = ['models', 'tmodels'],
                      help = 'Subdirectories (of temp directory) where the models should be stored')
  parser.add_argument('--zt-norm-directories', type = str, nargs = 5, dest='zt_dirs', 
                      default = ['zt_norm_A', 'zt_norm_B', 'zt_norm_C', 'zt_norm_D', 'zt_norm_D_sameValue'],
                      help = 'Subdiretories (of temp directory) where to write the zt_norm values')
  parser.add_argument('--score-dirs', type = str, nargs = 2, dest='score_dirs',
                      default = ['nonorm', 'ztnorm'],
                      help = 'Subdirectories (of user directories) where to write the results to')
  
  parser.add_argument('-x', '--grid', metavar = 'FILE', type = str, 
                      help = 'Configuration file for the grid setup')

  parser.add_argument('-n', '--preprocessing', type = int, nargs = 2,  
                      help = 'Perform image preprocessing on the given range of images')
  
  parser.add_argument('-e', '--feature-extraction-training', action='store_true', dest = 'training',
                      help = 'Perform feature extraction training')
  
  parser.add_argument('-f', '--feature-extraction', type = int, nargs = 2,
                      help = 'Perform feature extraction for the given range of preprocessed images')
  
  parser.add_argument('-m', '--model-generation', type = int, nargs = 2,
                      help = 'Generate the given range of models from the features')
  
  parser.add_argument('-s', '--compute-scores', type = int, nargs = 2,
                      help = 'Compute scores for the given range of models')
  
  parser.add_argument('-S', '--score-type', type = str, choices=['A', 'B', 'C', 'D', 'Z'],
                      help = 'The type of scores that should be computed')
  
  parser.add_argument('-g', '--group', type = str,
                      help = 'The group for which the current action should be performed')
  
  parser.add_argument('-z', '--zt-norm', action='store_false', dest = 'zt_norm',
                      help = 'DISABLE the computation of ZT norms')
  
  parser.add_argument('-r', '--force', action='store_true',
                      help = 'Force to erase former data if already exist')
  
  args = parser.parse_args()
  
  
  if not args.grid:
    default_tool_chain(args)
    
  

    
    

  

if __name__ == "__main__":
  main()
