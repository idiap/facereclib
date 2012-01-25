#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import os
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
      default=False, help='It is currently not possible to paralellize this script, and hence useless for the time being.')
  args = parser.parse_args()

  # Loads the configuration 
  import imp 
  config = imp.load_source('config', args.config_file)

  # Database
  db = config.db

  # Remove old file if required
  if args.force and os.path.exists(config.pca_model_filename):
    print "Removing old PCA model"
    os.remove(config.pca_model_filename)

  # Checks that the base directory for storing the PCA model exists
  utils.ensure_dir(os.path.dirname(config.pca_model_filename))

  if os.path.exists(config.pca_model_filename):
    print "PCA model already exists"
  else:
    print "Training PCA model"
    train_files = db.files(directory=config.features_dir, extension=config.features_ext,
                           protocol=config.protocol, groups='world', **config.world_options) 
    print "Number of training files: " + str(len(train_files))
    import pca
    pca.pca_train(train_files, config.pca_model_filename, config.pca_n_outputs)

if __name__ == "__main__": 
  main()
