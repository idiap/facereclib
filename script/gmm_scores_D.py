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
  # (sorted) list of T-Norm models
  tnorm_models_ids = sorted(db.Tmodels(protocol=config.protocol, groups=args.group))
  # Number of Z-Norm impostor samples
  n_zsamples = len(db.Zfiles(protocol=config.protocol, groups=args.group))

  # finally, if we are on a grid environment, just find what I have to process.
  zsamples_split_id = 0
  if args.grid:
    n_zsamples_splits = int(math.ceil(n_zsamples / float(config.N_MAX_PROBES_PER_JOB)))
    pos = int(os.environ['SGE_TASK_ID']) - 1
    if pos >= len(tnorm_models_ids) * n_zsamples_splits:
      raise RuntimeError, "Grid request for job %d on a setup with %d jobs" % \
          (pos, len(tnorm_models_ids) * n_zsamples_splits)
    tnorm_models_ids = [tnorm_models_ids[pos / n_zsamples_splits]]
    zsamples_split_id = pos % n_zsamples_splits

  # Gets the Z-Norm impostor sample list
  zfiles = db.Zobjects(directory=config.gmmstats_dir, extension=config.gmmstats_ext, protocol=config.protocol, groups=args.group)
  
  # If we are on a grid environment, just keep the required split of Z-norm impostor samples
  if args.grid:
    zfiles_g = utils.split_dictionary(zfiles, config.N_MAX_PROBES_PER_JOB)
    zfiles = zfiles_g[zsamples_split_id]

  # Checks that the base directories for storing the ZT-norm D matrices exist
  utils.ensure_dir(os.path.join(config.zt_norm_D_dir, args.group))
  utils.ensure_dir(os.path.join(config.zt_norm_D_sameValue_dir, args.group))

  # Computes the D and D_sameValue matrices (or a split of them)
  import gmm
  gmm.gmm_ztnorm_D(tnorm_models_ids, config.tnorm_models_dir, zfiles, config.ubm_filename, db, 
            config.zt_norm_D_dir, config.zt_norm_D_sameValue_dir, args.group, zsamples_split_id)

if __name__ == "__main__": 
  main()
