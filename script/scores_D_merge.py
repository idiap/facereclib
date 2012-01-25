#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import os, sys, tempfile, shutil, math
import bob
import numpy as np

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

  # Directories containing the images and the annotations
  db = config.db

  N_MAX_SPLITS = 9999 # zfill is done with 4 zeros
  for group in ['dev','eval']:
    # (sorted) list of T-Norm models
    tnorm_models_ids = sorted(db.Tmodels(protocol=config.protocol, groups=group))
    started = False
    for tmodel_id in tnorm_models_ids:
      tm_started = False
      for split_id in range(0,N_MAX_SPLITS):
        # Loads and concatenates into single arrays
        split_path_D = os.path.join(config.zt_norm_D_dir, group, str(tmodel_id) + "_" + str(split_id).zfill(4) + ".hdf5")
        split_path_Ds = os.path.join(config.zt_norm_D_sameValue_dir, group, str(tmodel_id) + "_" + str(split_id).zfill(4) + ".hdf5")
        if split_id == 0:
          if not os.path.exists(split_path_D):
            raise RuntimeError, "Cannot find file %s" % split_path_D
          if not os.path.exists(split_path_Ds):
            raise RuntimeError, "Cannot find file %s" % split_path_Ds
        elif not os.path.exists(split_path_D) and os.path.exists(split_path_Ds):
          raise RuntimeError, "Cannot find file %s" % split_path_D
        elif os.path.exists(split_path_D) and not os.path.exists(split_path_Ds):
          raise RuntimeError, "Cannot find file %s" % split_path_Ds
        elif not os.path.exists(split_path_D) and not os.path.exists(split_path_Ds):
          break
        D_tmp = bob.io.load(split_path_D)
        D_s_tmp = bob.io.load(split_path_Ds)
        if not tm_started:
          D_tm = D_tmp
          D_s_tm = D_s_tmp
          tm_started = True
        else:
          D_tm = np.concatenate((D_tm, D_tmp), 1)
          D_s_tm = np.concatenate((D_s_tm, D_s_tmp), 1)
      if not started:
        D = D_tm
        D_s = D_s_tm
        started = True
      else:
        D = np.concatenate((D, D_tm))
        D_s = np.concatenate((D_s, D_s_tm))
    # Saves to files
    bob.io.save(D, os.path.join(config.zt_norm_D_dir, group, "D.hdf5"))
    bob.io.save(D_s, os.path.join(config.zt_norm_D_sameValue_dir, group, "D_sameValue.hdf5"))

if __name__ == "__main__": 
  main()
