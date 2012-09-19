#!../bin/python

import subprocess
import os
import argparse

from .. import utils

all_algorithms = ('eigenface', 'lda', 'gaborgraph', 'lgbphs', 'gmm', 'isv', 'plda', 'bic')

def command_line_arguments():
  parser = argparse.ArgumentParser(description="Execute baseline algorithms with default parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-a', '--algorithms', choices = all_algorithms, default = ('eigenface',), nargs = '+', help = "Select one (or more) algorithms that you want to execute")
  parser.add_argument('--all', action = 'store_true', help = "Select all algorithms")
  parser.add_argument('-d', '--database', choices = ('atnt', 'banca', 'mobio', 'multipie', 'frgc', 'arface', 'gbu', 'lfw', 'xm2vts', 'scface'), default = 'atnt', help = "The database on which the baseline algorithm is executed")
  parser.add_argument('-p', '--protocol', default = 'None', help = "The protocol for the desired database")
  parser.add_argument('-s', '--share-preprocessing', action = 'store_true', help = "Share the preprocessed image directory?\nWARNING! When using this option and the --grid option, please let the first algorithm finish, until you start the next one")

  parser.add_argument('-g', '--grid', action = 'store_true', help = "Execute the algorithm in the SGE grid")
  parser.add_argument('-x', '--dry-run', action = 'store_true', help = "Just print the commands, but do not execute them")
  utils.add_logger_command_line_option(parser)

  parser.add_argument('-e', '--evaluate', action = 'store_true', help = "Evaluate the results of the algorithms (instead of running them)")

  parser.add_argument('parameters', nargs = argparse.REMAINDER, help = "Parameters directly passed to the face verification script.")

  args = parser.parse_args()
  if args.all:
    args.algorithms = all_algorithms

  utils.set_verbosity_level(args.verbose)

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

def plda():
  features      = "linearize.py"
  tool          = "pca+plda.py"
  grid          = "demanding.py"
  return (features, tool, grid)

def bic():
  features      = "linearize.py"
  tool          = "bic.py"
  grid          = "demanding.py"
  return (features, tool, grid)

faceverify_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
bin_dir = os.path.join(faceverify_dir, "bin")
config_dir = os.path.join(faceverify_dir, "config")
jman = os.path.join(bin_dir,'jman')
script = os.path.join(bin_dir, "faceverify_zt.py")


def available_protocols(database):
  configs = os.listdir(os.path.join(config_dir, 'database'))
  res=[]
  for file in configs:
    if file.find(database) == 0:
      parts = os.path.splitext(file)
      if parts[1] == '.py':
        res.append(parts[0][len(database)+1:])
  return res


def main():

  args = command_line_arguments()

  if args.protocol not in available_protocols(args.database):
    raise ValueError("The protocol '%s' for database '%s' is not (yet) available. Please choose one of %s"%(args.protocol, args.database, available_protocols(args.database)))

  # check the database configuration file
  database = os.path.join(config_dir, "database", args.database + "_" + args.protocol + ".py")
  has_zt_norm = args.database in ('banca', 'mobio', 'xm2vts')
  has_eval = args.database in ('banca', 'mobio', 'xm2vts', 'lfw')
  assert os.path.exists(database)

  # we always use Tan&Triggs preprocessing
  preprocessing = os.path.join(config_dir, "preprocessing", "tan_triggs.py")

  if args.evaluate: # evaluate the results

    # refresh if not yet done
    subprocess.call([jman, 'refresh'])
    utils.info("Cleaning up ...")
    subprocess.call([jman, 'delete', 'failure.db', '-rR'])
    subprocess.call([jman, 'delete', 'success.db', '-rR'])

    for algorithm in args.algorithms:

      utils.info("Evaluating algorithm '" + algorithm + "'")
      result_dir = os.path.join('/idiap/user', os.environ['USER'], args.database, 'baselines', algorithm, 'scores', args.protocol)

      folders = ('nonorm', 'ztnorm') if has_zt_norm else ('nonorm',)
      for dir in folders:
        dev_file = os.path.join(result_dir, dir, 'scores-dev')
        eval_file = os.path.join(result_dir, dir, 'scores-eval') if has_eval else dev_file
        if not os.path.exists(dev_file) or not os.path.exists(eval_file):
          utils.warn("The result file '%s' and/or '%s' does not exist," % (dev_file, eval_file))
          if os.path.exists("failure.db"):
            utils.warn("... and there were errors.")
          elif os.path.exists("submitted.db"):
            utils.warn("... maybe the jobs still run.")
          else:
            utils.warn("... although they should. Did you use some non-standard faceverify arguments?")
          continue

        call = [
                 os.path.join(bin_dir, 'bob_compute_perf.py'),
                 '-d', dev_file,
                 '-t', eval_file,
                 '-x'
               ]

        utils.info("Executing command:")
        utils.info(' '.join(call))

        if not args.dry_run:
          subprocess.call(call)


  else: # execution of the job is requested

    for algorithm in args.algorithms:

      utils.info("Executing algorithm '%s'" % algorithm)

      # get the setup for the desired algorithm
      setup = eval(algorithm)()
      features  = os.path.join(config_dir, "features", setup[0])
      tool      = os.path.join(config_dir, "tools", setup[1])
      grid      = os.path.join(config_dir, "grid", setup[2])
      if len(setup) > 3:
        preprocessing = os.path.join(config_dir, "preprocessing", setup[3])
        if args.share_preprocessing:
          utils.warn("Ignoring --share-preprocessing option for alogrithm '%s' since it requires a special setup" % algorithm)


      sub_directory = os.path.join("baselines", algorithm)
      # we share the preprocessed images if desired, so they don't have to be re-generated

      command = [
                  script,
                  "--database", database,
                  "--preprocessing", preprocessing,
                  "--features", features,
                  "--tool", tool,
                  '--sub-directory', sub_directory
                ]

      if args.grid:
        command.extend(['--grid', grid])

      if has_zt_norm:
        command.extend(['--zt-norm'])

      if has_eval:
        command.extend(['--groups', 'dev', 'eval'])

      if args.share_preprocessing and len(setup) == 3:
        command.extend(['--preprocessed-image-directory', '../preprocessed_images'])

      if args.parameters:
        command.extend(args.parameters[1:])

      if args.verbose:
        command.append("-" + "v"*args.verbose)

      utils.info("Executing command:")
      utils.info(' '.join(command))
      if not args.dry_run:
        subprocess.call(command)

if __name__ == "__main__":
  main()

