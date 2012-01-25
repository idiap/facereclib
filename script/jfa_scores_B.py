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
  parser.add_argument('-g', '--group', metavar='STR', type=str,
      dest='group', default="", help='Database group (\'dev\' or \'eval\') for which to retrieve models.')
  parser.add_argument('-c', '--config-file', metavar='FILE', type=str,
      dest='config_file', default="", help='Filename of the configuration file to use to run the script on the grid (defaults to "%(default)s")')
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
  # Number of Z-Norm impostor samples
  n_zsamples = len(db.Zfiles(protocol=config.protocol, groups=args.group))

  # finally, if we are on a grid environment, just find what I have to process.
  zsamples_split_id = 0
  if args.grid:
    n_zsamples_splits = int(math.ceil(n_zsamples / float(config.N_MAX_PROBES_PER_JOB)))
    pos = int(os.environ['SGE_TASK_ID']) - 1
    if pos >= len(models_ids) * n_zsamples_splits:
      raise RuntimeError, "Grid request for job %d on a setup with %d jobs" % \
          (pos, len(models_ids) * n_zsamples_splits)
    models_ids = [models_ids[pos / n_zsamples_splits]]
    zsamples_split_id = pos % n_zsamples_splits

  # Gets the Z-Norm impostor sample list
  zfiles = db.Zobjects(directory=config.gmmstats_dir, extension=config.gmmstats_ext, protocol=config.protocol, groups=args.group)
  
  # If we are on a grid environment, just keep the required split of Z-norm impostor samples
  if args.grid:
    zfiles_g = utils.split_dictionary(zfiles, config.N_MAX_PROBES_PER_JOB)
    zfiles = zfiles_g[zsamples_split_id]

  # Checks that the base directory for storing the ZT-norm B matrix exists
  utils.ensure_dir(os.path.join(config.zt_norm_B_dir, args.group))

  # Computes the B matrix (or a split of it)
  import jfa
  jfa.jfa_ztnorm_B(models_ids, config.models_dir, zfiles, config.jfabase_filename, config.ubm_filename, db, 
            config.zt_norm_B_dir, args.group, zsamples_split_id)

if __name__ == "__main__": 
  main()
