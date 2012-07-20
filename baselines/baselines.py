#!../bin/python

import subprocess
import os
import argparse

all_algorithms = ('eigenface', 'lda', 'gaborgraph', 'lgbphs', 'gmm', 'isv')

def command_line_arguments():
  parser = argparse.ArgumentParser(description="Execute baseline algorithms with default parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('-a', '--algorithms', choices = all_algorithms, default = 'eigenface', nargs = '+', help = "Select one (or more) algorithms that you want to execute")
  parser.add_argument('--all', action = 'store_true', help = "Select all algorithms")
  parser.add_argument('-d', '--database', choices = ('banca', 'mobio', 'frgc', 'arface', 'GBU', 'LFW', 'xm2vts', 'scface'), default = 'banca', help = "The database on which the baseline algorithm is executed")
  parser.add_argument('-p', '--protocol', default = 'P', help = "The protocol for the desired database")
  parser.add_argument('-s', '--share-preprocessing', action = 'store_true', help = "Share the preprocessed image directory?\nWARNING! When using this option and the --grid option, please let the first algorithm finish, until you start the next one")
  
  parser.add_argument('-g', '--grid', action = 'store_true', help = "Execute the algorithm in the SGE grid")
  parser.add_argument('-x', '--dry-run', action = 'store_true', help = "Just print the commands, but do not execute them")
  
  parser.add_argument('-e', '--evaluate', action = 'store_true', help = "Evaluate the results of the algorithms (instead of running them)")
  parser.add_argument('-b', '--bob-install-directory', default = '/idiap/group/torch5spro/releases/bob-1.0.5/install/linux-x86_64-release', help = "The installation directory of Bob (needed for evaluation)")
  
#  parameter_group = parser.add_argument_group('Parameters for the underlying face verification script')
  parser.add_argument('parameters', nargs = argparse.REMAINDER, help = "Parameters directly passed to the face verification script.")

  args = parser.parse_args()
  if args.all:
    args.algorithms = all_algorithms
  
  return args

def eigenface():
  """Simple eigenface comparison"""
  features      = "linearize.py"
  tool          = "pca.py"
  grid          = "grid.py"
  return (features, tool, grid)
  
def lda():
  """LDA on eigenface features"""
  features      = "eigenfaces.py"
  tool          = "lda.py"
  grid          = "grid.py"
  return (features, tool, grid)
  
def gaborgraph():
  features      = "grid_graph.py"
  tool          = "gabor_jet.py"
  grid          = "grid.py"
  return (features, tool, grid)
  
def lgbphs():
  features      = "lgbphs.py"
  tool          = "lgbphs.py"
  grid          = "grid.py"
  preprocessing = "tan_triggs_with_offset.py"
  return (features, tool, grid, preprocessing)

def gmm():
  features      = "dct_blocks.py"
  tool          = "ubm_gmm.py"
  grid          = "demanding.py"
  return (features, tool, grid)
  
def isv():
  features      = "dct_blocks.py"
  tool          = "isv.py"
  grid          = "demanding.py"
  return (features, tool, grid)
  

if __name__ == "__main__":

  script = "../bin/faceverify_zt.py"
  config_dir = "../config"
  
  args = command_line_arguments()

  # check the database configuration file  
  database = os.path.join(config_dir, "database", args.database + "_" + args.protocol + ".py")
  has_zt_norm = args.database in ('banca', 'mobio', 'xm2vts')
  has_eval = args.database in ('banca', 'mobio', 'xm2vts', 'LFW')
  if not os.path.exists(database):
    raise InvalidArgumentException("The desired configuration file '" + os.path.realpath(database) + "' of the database '" + args.database + "' does not exist.")
  
  # we always use Tan&Triggs preprocessing
  preprocessing = os.path.join(config_dir, "preprocessing", "tan_triggs.py")

  if args.evaluate: # evaluate the results

    # refresh if not yet done      
    subprocess.call(['../bin/jman', 'refresh'])
    print "Cleaning up..."
    subprocess.call(['../bin/jman', 'delete', 'failure.db', '-rR'])
    subprocess.call(['../bin/jman', 'delete', 'success.db', '-rR'])

    for algorithm in args.algorithms:
  
      print "Evaluating algorithm '" + algorithm + "'"
      result_dir = os.path.join('/idiap/user', os.environ['USER'], args.database, 'baselines', algorithm, 'scores', args.protocol)

      folders = ('nonorm', 'ztnorm') if has_zt_norm else ('nonorm',)
      for dir in folders:
        dev_file = os.path.join(result_dir, dir, 'scores-dev')
        eval_file = os.path.join(result_dir, dir, 'scores-eval') if has_eval else dev_file
        if not os.path.exists(dev_file) or not os.path.exists(eval_file):
          print "The result file '" + dev_file + "' and/or '" + eval_file + "' does not exist,",
          if os.path.exists("failure.db"):
            print "and there were errors."
          elif os.path.exists("submitted.db"):
            print "maybe the jobs still run."
          else:
            print "although they should. Did you use some non-standard faceverify arguments?"
          continue
            
        call = [
                 os.path.join(args.bob_install_directory, 'bin', 'bob_compute_perf.py'),
                 '-d', dev_file,
                 '-t', eval_file,
                 '-x'
               ]
      
        print ' '.join(call)
      
        if not args.dry_run:
          subprocess.call(call)
        
      
  else: # execution of the job is requested

    for algorithm in args.algorithms:
    
      print "Executing algorithm '" + algorithm + "'"

      # get the setup for the desired algorithm
      setup = eval(algorithm)()
      features  = os.path.join(config_dir, "features", setup[0])
      tool      = os.path.join(config_dir, "tools", setup[1])
      grid      = os.path.join(config_dir, "grid", setup[2])
      if len(setup) > 3:
        preprocessing = os.path.join(config_dir, "preprocessing", setup[3])
    
    
      sub_directory = os.path.join("baselines", algorithm)
      # we share the preprocessed images if desired, so they don't have to be re-generated
    
      command = [
                  script, 
                  "--database", database, 
                  "--preprocessing", preprocessing,
                  "--features", features,
                  "--tool-chain", tool,
                  '--sub-directory', sub_directory
                ]
                
      if args.grid:
        command.extend(['--grid', grid])
      
      if not has_zt_norm:
        command.extend(['--no-zt-norm'])
        
      if not has_eval:
        command.extend(['--groups', 'dev'])
        
      if args.share_preprocessing and len(setup) == 3:
        command.extend(['--preprocessed-image-directory', '../preprocessed_images'])
                
      if args.parameters:
        command.extend(args.parameters[1:])
               
      print ' '.join(command)
      if not args.dry_run:
        subprocess.call(command)
