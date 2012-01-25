#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import os, math
import bob
import utils

import argparse

def main():

  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-c', '--config-file', metavar='FILE', type=str,
      dest='config_file', default="", help='Filename of the configuration file to use to run the script on the grid (defaults to "%(default)s")')
  parser.add_argument('-f', '--force', dest='force', action='store_true',
      default=False, help='Force to erase former data if already exist')
  parser.add_argument('--grid', dest='grid', action='store_true',
      default=False, help='If set, assumes it is being run using a parametric grid job. It orders all ids to be processed and picks the one at the position given by ${SGE_TASK_ID}-1')
  args = parser.parse_args()

  # Loads the configuration 
  import imp 
  config = imp.load_source('config', args.config_file)

  # Directories containing the images and the annotations
  db = config.db
  features_input = config.db.files(directory=config.features_dir, extension=config.features_ext, protocol=config.protocol, **config.all_files_options)
  gmmstats_output = config.db.files(directory=config.gmmstats_dir, extension=config.gmmstats_ext, protocol=config.protocol, **config.all_files_options)

  # finally, if we are on a grid environment, just find what I have to process.
  if args.grid:
    pos = int(os.environ['SGE_TASK_ID']) - 1
    n_jobs = int(math.ceil(len(features_input) / float(config.N_MAX_FILES_PER_JOB)))
    
    if pos >= n_jobs:
      raise RuntimeError, "Grid request for job %d on a setup with %d jobs" % \
          (pos, n_jobs)
    features_input_g = utils.split_dictionary(features_input, config.N_MAX_FILES_PER_JOB)[pos]
    features_input = features_input_g


  # Checks that the base directory for storing the gmm statistics exists
  utils.ensure_dir(config.gmmstats_dir)

  # Processes the features
  import gmm
  gmm.gmm_stats(features_input, config.ubm_filename, gmmstats_output, args.force)

if __name__ == "__main__": 
  main()
