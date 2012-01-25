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

  N_MAX_SPLITS = 9999 # zfill is done with 4 zeros
  for group in ['dev','eval']:
    # (sorted) list of models
    models_ids = sorted(db.models(protocol=config.protocol, groups=group))

    f = open(os.path.join(config.scores_nonorm_dir, "scores-" + group), 'w')
    # Concatenates the scores
    for model_id in models_ids:
      for split_id in range(0,N_MAX_SPLITS): 
        # Loads and concatenates
        split_path = os.path.join(config.scores_nonorm_dir, group, str(model_id) + "_" + str(split_id).zfill(4) + ".txt")
        if split_id == 0 and not os.path.exists(split_path):
          raise RuntimeError, "Cannot find file %s" % split_path
        elif not os.path.exists(split_path):
          break
        f.write(open(split_path, 'r').read())
    f.close()

    if config.zt_norm:
      # Checks that the base directory for storing the ZT-norm scores exists
      utils.ensure_dir(os.path.join(config.scores_ztnorm_dir, group))

      f = open(os.path.join(config.scores_ztnorm_dir, "scores-" + group), 'w')
      # Concatenates the scores
      for model_id in models_ids:
        f.write(open(os.path.join(config.scores_ztnorm_dir, group, str(model_id) + ".txt"), 'r').read())
      f.close()
    

if __name__ == "__main__": 
  main()
