#!/usr/bin/env python

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

  # Directories containing the features and the output projected features
  input_features = db.files(directory=config.features_dir, extension=config.features_ext, protocol=config.protocol, **config.all_files_options)
  output_features = db.files(directory=config.featuresProjected_dir, extension=config.featuresProjected_ext, protocol= config.protocol, **config.all_files_options)

  # finally, if we are on a grid environment, just find what I have to process.
  if args.grid:
    pos = int(os.environ['SGE_TASK_ID']) - 1
    n_total = config.TOTAL_ARRAY_JOBS
    n_per_job = math.ceil(len(input_features) / float(config.TOTAL_ARRAY_JOBS))
    
    if pos >= n_total:
      raise RuntimeError, "Grid request for job %d on a setup with %d jobs" % \
          (pos, n_total)
    input_features_g = utils.split_dictionary(input_features, n_per_job)[pos]
    input_features = input_features_g

  # Checks that the base directory for storing the features exists
  utils.ensure_dir(config.featuresProjected_dir)

  import pca
  pca.pca_project(input_features, config.pca_model_filename, output_features, args.force)

if __name__ == "__main__": 
  main()
