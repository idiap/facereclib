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
    # (sorted) list of models
    models_ids = sorted(db.models(protocol=config.protocol, groups=group))
    for tc in models_ids:
      started = False
      for split_id in range(0,N_MAX_SPLITS):
        # Loads and concatenates into single arrays
        split_path = os.path.join(config.zt_norm_A_dir, group, str(tc) + "_" + str(split_id).zfill(4) + ".hdf5")
        if split_id == 0 and not os.path.exists(split_path):
          raise RuntimeError, "Cannot find file %s" % split_path
        elif not os.path.exists(split_path):
          break
        A_tc = bob.io.load(split_path)
        if started == False:
          A = A_tc
          started = True
        else:
          A = np.concatenate((A, A_tc), 1) #TODO: TO BE CHECKED
      # Saves to files
      bob.io.save(A, os.path.join(config.zt_norm_A_dir, group, str(tc) + ".hdf5"))

if __name__ == "__main__": 
  main()
