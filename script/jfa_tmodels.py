#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import os, sys
import bob
import utils

import argparse

def main():

  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-g', '--group', metavar='STR', type=str,
      dest='group', default="", help='Database group (\'dev\' or \'eval\') for which to retrieve T-Norm models.')
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

  # Database
  db = config.db
  # (sorted) list of T-Norm models
  tnorm_models_ids = sorted(db.Tmodels(protocol=config.protocol, groups=args.group))

  # finally, if we are on a grid environment, just find what I have to process.
  if args.grid:
    pos = int(os.environ['SGE_TASK_ID']) - 1
    if pos >= len(tnorm_models_ids):
      raise RuntimeError, "Grid request for job %d on a setup with %d jobs" % \
          (pos, len(tnorm_models_ids))
    tnorm_models_ids = [tnorm_models_ids[pos]]

  # Checks that the base directory for storing the T-Norm models exists
  utils.ensure_dir(config.tnorm_models_dir)

  # Trains the T-Norm models
  import jfa
  for tmodel_id in tnorm_models_ids:
    # Path to the T-Norm model
    tmodel_path = os.path.join(config.tnorm_models_dir, str(tmodel_id) + ".hdf5")

    enrol_files = db.Tfiles(directory=config.gmmstats_dir, extension=config.gmmstats_ext, 
                            groups=args.group, protocol=config.protocol, model_ids=(tmodel_id,))
    jfa.jfa_enrol_model(enrol_files, tmodel_path, config.jfabase_filename, config.ubm_filename,
                config.n_iter_enrol, args.force)

if __name__ == "__main__": 
  main()
