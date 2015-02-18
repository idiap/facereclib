#!../bin/python
from __future__ import print_function

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

# collect all algorithms that we provide baselines for
all_algorithms = ['dummy', 'eigenface', 'lda', 'gaborgraph', 'lgbphs', 'gmm', 'isv', 'plda', 'bic']
try:
  # try if the CSU extension is enabled
  utils.tests.load_resource('lrpca', 'tool')
  utils.tests.load_resource('lda-ir', 'tool')
  all_algorithms += ['lrpca', 'lda_ir']
except:
  pass


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
  # - the database to choose
  parser.add_argument('-b', '--baseline-directory', default = 'baselines', help = 'The sub-directory, where the baseline results are stored.')
  # - the directory to write
  parser.add_argument('-f', '--directory', help = 'The directory to write the data of the experiment into. If not specified, the default directories of the faceverify script are used (see bin/faceverify.py --help).')
  # - special option to share preprocessing. This can be used to save some time.
  parser.add_argument('-s', '--share-preprocessing', action = 'store_true', help = 'Share the preprocessed data directory?\nWARNING! When using this option and the --grid option, please let the first algorithm finish, until you start the next one.')

  # - use the Idiap grid -- option is only useful if you are at Idiap
  parser.add_argument('-g', '--grid', action = 'store_true', help = 'Execute the algorithm in the SGE grid.')
  # - run in parallel on the local machine
  parser.add_argument('-p', '--parallel', type=int, help = 'Run the algorithms in parallel on the local machine, using the given number of parallel threads')
  # - perform ZT-normalization
  parser.add_argument('-z', '--zt-norm', action = 'store_true', help = 'Compute the ZT norm for the files.')

  # - just print?
  parser.add_argument('-q', '--dry-run', action = 'store_true', help = 'Just print the commands, but do not execute them.')

  # - evaluate the algorithm (after it has finished)
  parser.add_argument('-e', '--evaluate', nargs='+', choices = ('EER', 'HTER', 'ROC', 'DET', 'CMC', 'RR'), help = 'Evaluate the results of the algorithms (instead of running them) using the given evaluation techniques.')

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


def dummy():
  """Dummy script just for testing the tool chain"""
  import pkg_resources
  features      = 'eigenfaces'
  tool          = pkg_resources.resource_filename('facereclib', 'configurations/tools/dummy.py')
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
  return (features, tool, grid)

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
  features      = 'grid-graph'
  tool          = 'bic-jets'
  grid          = 'grid'
  preprocessing = 'face-crop'
  return (features, tool, grid, preprocessing)

def lrpca():
  """Local Region PCA"""
  features      = 'lrpca'
  tool          = 'lrpca'
  grid          = 'grid'
  preprocessing = 'lrpca'
  return (features, tool, grid, preprocessing)

def lda_ir():
  """LDA-IR (a.k.a. CohortLDA)"""
  features      = 'lda-ir'
  tool          = 'lda-ir'
  grid          = 'grid'
  preprocessing = 'lda-ir'
  return (features, tool, grid, preprocessing)



def main(command_line_parameters = sys.argv):

  # Collect command line arguments
  args = command_line_arguments(command_line_parameters[1:])

  # Check the database configuration file
  has_zt_norm = args.database in ('banca', 'mobio', 'multipie', 'scface')
  has_eval = args.database in ('banca', 'mobio', 'multipie', 'scface', 'xm2vts')

  if args.evaluate:
    # call the evaluate script with the desired parameters

    # get the base directory of the results
    base_dir = args.directory if args.directory else "results"
    if not os.path.exists(base_dir):
      if not args.dry_run:
        raise IOError("The result directory cannot be found. Please specify the --directory as it was specified during execution of the algorithms.")

    # get the result directory of the database
    result_dir = os.path.join(base_dir, args.database, args.baseline_directory)
    if not os.path.exists(result_dir):
      if not args.dry_run:
        raise IOError("The result directory for the desired database cannot be found. Did you already run the experiments for this database?")

    # iterate over the algorithms and collect the result files
    result_dev = []
    result_eval = []
    result_zt_dev = []
    result_zt_eval = []
    legends = []

    # evaluate the results
    for algorithm in args.algorithms:
      nonorm_sub_dir = os.path.join(algorithm, 'scores')
      if not os.path.exists(os.path.join(result_dir, nonorm_sub_dir)):
        utils.warn("Skipping algorithm '%s' since the results cannot be found." % algorithm)
        continue
      protocols = os.listdir(os.path.join(result_dir, nonorm_sub_dir))
      if not len(protocols):
        utils.warn("Skipping algorithm '%s' since the results cannot be found."%algorithm)
        continue
      if len(protocols) > 1:
        utils.warn("There are several protocols found in directory '%s'. Here, we use protocol '%s'." %(os.path.join(result_dir, nonorm_sub_dir), protocols[0]))

      nonorm_sub_dir = os.path.join(nonorm_sub_dir, protocols[0], 'nonorm')
      ztnorm_sub_dir = os.path.join(nonorm_sub_dir, protocols[0], 'ztnorm')

      # collect the resulting files
      if os.path.exists(os.path.join(result_dir, nonorm_sub_dir, 'scores-dev')):
        result_dev.append(os.path.join(nonorm_sub_dir, 'scores-dev'))
        legends.append(algorithm)
      if has_eval and os.path.exists(os.path.join(result_dir, nonorm_sub_dir, 'scores-eval')):
        result_eval.append(os.path.join(nonorm_sub_dir, 'scores-eval'))

      if has_zt_norm:
        if os.path.exists(os.path.join(result_dir, ztnorm_sub_dir, 'scores-dev')):
          result_zt_dev.append(os.path.join(ztnorm_sub_dir, 'scores-dev'))
        if has_eval and os.path.exists(os.path.join(result_dir, ztnorm_sub_dir, 'scores-eval')):
          result_zt_eval.append(os.path.join(ztnorm_sub_dir, 'scores-eval'))

    # check if we have found some results
    if not result_dev:
      utils.warn("No result files were detected -- skipping evaluation.")
      return

    # call the evaluate script
    base_call = ['./bin/evaluate.py', '--directory', result_dir, '--legends'] + legends
    if 'EER' in args.evaluate:
      base_call += ['--criterion', 'EER']
    elif 'HTER' in args.evaluate:
      base_call += ['--criterion', 'HTER']
    if 'ROC' in args.evaluate:
      base_call += ['--roc', 'ROCxxx.pdf']
    if 'DET' in args.evaluate:
      base_call += ['--det', 'DETxxx.pdf']
    if 'CMC' in args.evaluate:
      base_call += ['--cmc', 'CMCxxx.pdf']
    if 'RR' in args.evaluate:
      base_call += ['--rr']
    if args.verbose:
      base_call += ['-' + 'v'*args.verbose]

    # first, run the nonorm evaluation
    if result_zt_dev:
      call = [command.replace('xxx','_dev') for command in base_call]
    else:
      call = [command.replace('xxx','') for command in base_call]
    call += ['--dev-files'] + result_dev
    if result_eval:
      call += ['--eval-files'] + result_eval

    utils.info("Executing command:")
    print(" ".join(call))
    if not args.dry_run:
      subprocess.call(call)

    # now, also run the ZT norm evaluation
    if result_zt_dev:
      call = [command.replace('xxx','_eval') for command in base_call]
      call += ['--dev-files'] + result_zt_dev
      if result_eval:
        call += ['--eval-files'] + result_zt_eval

      utils.info("Executing command:")
      print(" ".join(call))
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
      sub_directory = os.path.join(args.baseline_directory, algorithm)

      # create the command to the faceverify script
      command = [
                  './bin/faceverify.py',
                  '--database', args.database,
                  '--preprocessing', preprocessing,
                  '--features', features,
                  '--tool', tool,
                  '--sub-directory', sub_directory
                ]

      # add grid argument, if available
      if args.grid:
        command.extend(['--grid', grid])

      if args.parallel:
        command.extend(['--grid', 'facereclib.utils.GridParameters("local",number_of_parallel_processes=%d)'%args.parallel, '--run-local-scheduler'])

      # compute ZT-norm if the database provides this setup
      if has_zt_norm and args.zt_norm:
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
      print (' '.join(command))

      # run the command
      if not args.dry_run:
        subprocess.call(command)


if __name__ == "__main__":
  main()
