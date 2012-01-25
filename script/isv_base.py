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

  # Checks that the base directory for storing the ubm exists
  utils.ensure_dir(os.path.dirname(config.jfabase_filename))

  clients = db.clients(protocol=config.protocol, groups="world", **config.jfa_options_clients)
  train_files = []
  n_files = 0
  for c in clients:
    clist = db.files(directory=config.gmmstats_dir, extension=config.gmmstats_ext, protocol=config.protocol, groups="world", model_ids=(c,), **config.jfa_options_files)
    train_files.append(clist)
    n_files += len(clist)
  print "Number of training files: " + str(n_files)
  import jfa
  jfa.isv_train_base(train_files, config.jfabase_filename, config.ubm_filename, 
                     config.ru, config.n_iter_train, config.relevance_factor, args.force)

if __name__ == "__main__": 
  main()
