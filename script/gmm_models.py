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
      dest='group', default="", help='Database group (\'dev\' or \'eval\') for which to retrieve models.')
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
  # (sorted) list of models
  models_ids = sorted(db.models(protocol=config.protocol, groups=args.group))

  # finally, if we are on a grid environment, just find what I have to process.
  if args.grid:
    pos = int(os.environ['SGE_TASK_ID']) - 1
    if pos >= len(models_ids):
      raise RuntimeError, "Grid request for job %d on a setup with %d jobs" % \
          (pos, len(models_ids))
    models_ids = [models_ids[pos]]

  # Checks that the base directory for storing the T-Norm models exists
  utils.ensure_dir(config.models_dir)

  # Trains the models
  import gmm
  for model_id in models_ids:
    # Path to the model
    model_path = os.path.join(config.models_dir, str(model_id) + ".hdf5")

    # Removes old file if required
    if args.force and os.path.exists(model_path):
      print "Removing old GMM model"
      os.remove(model_path)

    if os.path.exists(model_path):
      print "Model %s already exists." % model_path
    else:
      print "Enroling model %s." % model_path
      enrol_files = db.files(directory=config.features_dir, extension=config.features_ext, 
                             protocol=config.protocol, model_ids=(model_id,), purposes='enrol')
      gmm.gmm_enrol_model(enrol_files, model_path, config.ubm_filename,
                config.iterg_enrol, config.convergence_threshold, config.variance_threshold,
                config.relevance_factor, config.responsibilities_threshold)

if __name__ == "__main__": 
  main()
