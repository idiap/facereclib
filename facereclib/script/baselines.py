#!../bin/python

import subprocess
import os
import sys
import argparse

from .. import utils

# This is the default set of algorithms that can be run using this script.
all_databases = utils.resources.resource_keys('database')
# check, which databases can actually be assessed
available_databases = []
for database in all_databases:
  try:
    utils.tests.load_resource(database, 'database')
    available_databases.append(database)
  except:
    pass

all_algorithms = ('dummy', 'eigenface', 'lda', 'gaborgraph', 'lgbphs', 'gmm', 'isv', 'plda', 'bic')

def command_line_arguments(command_line_parameters):
  """Defines the command line parameters that are accepted."""

  # create parser
  parser = argparse.ArgumentParser(description='Execute baseline algorithms with default parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # add parameters
  # - the algorithm to execute
  parser.add_argument('-a', '--algorithms', choices = all_algorithms, default = ('eigenface',), nargs = '+', help = 'Select one (or more) algorithms that you want to execute.')
  parser.add_argument('--all', action = 'store_true', help = 'Select all algorithms.')
  # - the database to choose
  parser.add_argument('-d', '--database', choices = available_databases, default = 'atnt', help = 'The database on which the baseline algorithm is executed.')
  # - the directory to write
  parser.add_argument('-f', '--directory', help = 'The directory to write the data of the experiment into. If not specified, the default directories of the faceverify script are used (see bin/faceverify.py --help).')
  # - special option to share preprocessing. This can be used to save some time.
  parser.add_argument('-s', '--share-preprocessing', action = 'store_true', help = 'Share the preprocessed data directory?\nWARNING! When using this option and the --grid option, please let the first algorithm finish, until you start the next one.')

  # - use the Idiap grid -- option is only useful if you are at Idiap
  parser.add_argument('-g', '--grid', action = 'store_true', help = 'Execute the algorithm in the SGE grid.')

  # - just print?
  parser.add_argument('-q', '--dry-run', action = 'store_true', help = 'Just print the commands, but do not execute them.')

  # - evaluate the algorithm (after it has finished)
  parser.add_argument('-e', '--evaluate', action = 'store_true', help = 'Evaluate the results of the algorithms (instead of running them).')

  # - other parameters that are passed to the underlying script
  parser.add_argument('parameters', nargs = argparse.REMAINDER, help = 'Parameters directly passed to the face verification script.')

  utils.add_logger_command_line_option(parser)
  args = parser.parse_args(command_line_parameters)
  if args.all:
    args.algorithms = all_algorithms[1:]

  utils.set_verbosity_level(args.verbose)

  return args


# In these functions, some default experiments are prepared.
# An experiment consists of three configuration files:
# - The features to be extracted
# - The algorithm to be run
# - The grid configuration that it requires (only used when the --grid option is chosen)

# Some default variables that are required
faceverify_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
bin_dir = os.path.join(faceverify_dir, 'bin')
config_dir = os.path.join(faceverify_dir, 'facereclib/configurations')
script = os.path.join(bin_dir, 'faceverify.py')


def dummy():
  """Dummy script just for testing the tool chain"""
  features      = 'eigenfaces'
  tool          = os.path.join(config_dir, 'tools', 'dummy.py')
  grid          = 'grid'
  return (features, tool, grid)

def eigenface():
  """Simple eigenface comparison"""
  features      = 'linearize'
  tool          = 'pca'
  grid          = 'grid'
  preprocessing = 'face-crop'
  return (features, tool, grid, preprocessing)

def lda():
  """LDA on eigenface features"""
  features      = 'eigenfaces'
  tool          = 'lda'
  grid          = 'grid'
  preprocessing = 'face-crop'
  return (features, tool, grid, preprocessing)

def gaborgraph():
  """Gabor grid graphs using a Gabor phase based similarity measure"""
  features      = 'grid-graph'
  tool          = 'gabor-jet'
  grid          = 'grid'
  return (features, tool, grid)

def lgbphs():
  """Local Gabor binary pattern histogram sequences"""
  features      = 'lgbphs'
  tool          = 'lgbphs'
  grid          = 'grid'
  preprocessing = os.path.join(config_dir, 'preprocessing', 'tan_triggs_with_offset.py')
  return (features, tool, grid, preprocessing)

def gmm():
  """UBM/GMM modelling of DCT block features"""
  features      = 'dct'
  tool          = 'gmm'
  grid          = 'demanding'
  return (features, tool, grid)

def isv():
  """Inter-Session-Variability modelling of DCT block features"""
  features      = 'dct'
  tool          = 'isv'
  grid          = 'demanding'
  return (features, tool, grid)

def plda():
  """Probabilistic LDA using PCA+PLDA on pixel-based features"""
  features      = 'linearize'
  tool          = 'pca+plda'
  grid          = 'demanding'
  return (features, tool, grid)

def bic():
  """The Bayesian Intrapersonal/Extrapersonal classifier"""
  features      = 'linearize'
  tool          = 'bic'
  grid          = 'demanding'
  return (features, tool, grid)



def main(command_line_parameters = sys.argv):

  # Collect command line arguments
  args = command_line_arguments(command_line_parameters[1:])

  # Check the database configuration file
  has_zt_norm = args.database in ('banca', 'mobio-male', 'mobio-female', 'multipie-U', 'multipie-P', 'scface')
  has_eval = args.database in ('banca', 'mobio-male', 'mobio-female', 'multipie-U', 'multipie-P', 'scface', 'xm2vts', 'lfw')

  if args.evaluate:
    # evaluate the results
    for algorithm in args.algorithms:

      utils.info("Evaluating algorithm '" + algorithm + "'")

      database = utils.resources.load_resource(args.database, 'database')

      if args.directory:
        # append the directories
        result_dir = os.path.join(args.directory, args.database, 'baselines', algorithm, 'scores', database.protocol)
      else:
        # this is the default result directory
        result_dir = os.path.join('/idiap/user', os.environ['USER'], args.database, 'baselines', algorithm, 'scores', database.protocol)

      # sub-directories of the result directories that contain the score files
      folders = ('nonorm', 'ztnorm') if has_zt_norm else ('nonorm',)

      for dir in folders:
        # score files
        dev_file = os.path.join(result_dir, dir, 'scores-dev')
        eval_file = os.path.join(result_dir, dir, 'scores-eval') if has_eval else dev_file

        # check if the score files are already there (i.e., the experiments have finished)
        if not args.dry_run and (not os.path.exists(dev_file) or not os.path.exists(eval_file)):
          utils.error("The result file '%s' and/or '%s' does not exist" % (dev_file, eval_file))
          continue

        # generate a call to a bob function to do the actual evaluation
        call = [
                 os.path.join(bin_dir, 'bob_compute_perf.py'),
                 '-d', dev_file,
                 '-t', eval_file,
                 '-x'
               ]

        # print the command so that it can be re-issued on need
        utils.info("Executing command:")
        print ' '.join(call)

        # execute the command
        if not args.dry_run:
          subprocess.call(call)

  else:

    # execution of the job is requested
    for algorithm in args.algorithms:

      utils.info("Executing algorithm '%s'" % algorithm)

      # get the setup for the desired algorithm
      setup = eval(algorithm)()
      features  = setup[0]
      tool      = setup[1]
      grid      = setup[2]
      if len(setup) > 3:
        preprocessing = setup[3]
        if args.share_preprocessing:
          utils.warn("Ignoring --share-preprocessing option for algorithm '%s' since it requires a special setup" % algorithm)
      else:
        # by default, we use Tan & Triggs preprocessing
        preprocessing = 'tan-triggs'

      # this is the default sub-directory that is used
      sub_directory = os.path.join('baselines', algorithm)

      # create the command to the faceverify script
      command = [
                  script,
                  '--database', args.database,
                  '--preprocessing', preprocessing,
                  '--features', features,
                  '--tool', tool,
                  '--sub-directory', sub_directory
                ]

      # add grid argument, if available
      if args.grid:
        command.extend(['--grid', grid])

      # compute ZT-norm if the database provides this setup
      if has_zt_norm:
        command.extend(['--zt-norm'])

      # compute results for both 'dev' and 'eval' group if the database provides these
      if has_eval:
        command.extend(['--groups', 'dev', 'eval'])

      # we share the preprocessed data if desired, so they don't have to be re-generated
      if args.share_preprocessing and len(setup) == 3:
        command.extend(['--preprocessed-data-directory', '../preprocessed_data'])

      # set the directories, if desired; we set both directories to be identical.
      if args.directory:
        command.extend(['--temp-directory', os.path.join(args.directory, args.database), '--result-directory', os.path.join(args.directory, args.database)])

      # set the verbosity level
      if args.verbose:
        command.append('-' + 'v'*args.verbose)

      # add the command line arguments that were specified on command line
      if args.parameters:
        command.extend(args.parameters[1:])

      # print the command so that it can easily be re-issued
      utils.info("Executing command:")
      print ' '.join(command)

      # run the command
      if not args.dry_run:
        subprocess.call(command)

if __name__ == "__main__":
  main()

