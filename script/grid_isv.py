#!/usr/bin/env python
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""Submits all feature creation jobs to the Idiap grid"""

import os, sys, math
import argparse

def checked_directory(base, name):
  """Checks and returns the directory composed of os.path.join(base, name). If
  the directory does not exist, raise a RuntimeError.
  """
  retval = os.path.join(base, name)
  if not os.path.exists(retval):
    raise RuntimeError, "You have not created a link to '%s' at your '%s' installation - you don't have to, but then you need to edit this script to eliminate this error" % (name, base)
  return retval

# Finds myself first
FACERECLIB_DIR = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

# Defines the gridtk installation root - by default we look at a fixed location
# in the currently detected FACERECLIB_DIR. You can change this and hard-code
# whatever you prefer.
GRIDTK_DIR = checked_directory(FACERECLIB_DIR, 'gridtk')
sys.path.insert(0, GRIDTK_DIR)

# Defines the bob installation root - by default we look at a fixed
# location in the currently detected FACERECLIB_DIR. You can change this and
# hard-code whatever you prefer.
#BOB_DIR = checked_directory(FACERECLIB_DIR, 'bob')

# The wrapper is required to bracket the execution environment for the facereclib
# scripts:
FACERECLIB_WRAPPER = os.path.join(FACERECLIB_DIR, 'shell.py')

# The environment assures the correct execution of the wrapper and the correct
# location of both the 'facereclib' and 'bob' packages.
FACERECLIB_WRAPPER_ENVIRONMENT = [
    'FACERECLIB_DIR=%s' % FACERECLIB_DIR
#    'BOB_DIR=%s' % BOB_DIR,
    ]

def submit(job_manager, command, dependencies=[], array=None, queue=None, mem=None, hostname=None, pe_opt=None):
  """Submits one job using our specialized shell wrapper. We hard-code certain
  parameters we like to use. You can change general submission parameters
  directly at this method."""
 
  from gridtk.tools import make_python_wrapper, random_logdir
  name = os.path.splitext(os.path.basename(command[0]))[0]
  logdir = os.path.join('logs', random_logdir())

  use_cmd = make_python_wrapper(FACERECLIB_WRAPPER, command)
  return job_manager.submit(use_cmd, deps=dependencies, cwd=True,
      queue=queue, mem=mem, hostname=hostname, pe_opt=pe_opt,
      stdout=logdir, stderr=logdir, name=name, array=array, 
      env=FACERECLIB_WRAPPER_ENVIRONMENT)

def main():
  """The main entry point, control here the jobs options and other details"""

  # Parses options
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-c', '--config-file', metavar='FILE', type=str,
      dest='config_file', default="", help='Filename of the configuration file to use to run the script on the grid (defaults to "%(default)s")')
  parser.add_argument('-j', '--nogrid', dest='nogrid', action='store_true',
      default=False, help='Do not use the SGE grid, but the current machine directly.')
  args = parser.parse_args()

  # Loads the configuration 
  import imp
  config = imp.load_source('config', args.config_file)

  # Let's create the job manager
  from gridtk.manager import JobManager
  jm = JobManager()

  # Database
  db = config.db
  
  # Trains the UBM
  cmd_ubm = [ 
              'gmm_ubm.py', 
              '--config-file=%s' % args.config_file, 
              '--grid'
            ]
  if args.nogrid: subprocess.call(cmd_ubm)
  else:
    job_ubm = submit(jm, cmd_ubm, dependencies=[], array=None, queue='q1d', mem='4G')
    print 'submitted:', job_ubm

  # Computes the GMM Stats if linear scoring is performed
  job_gmmstats = []
  n_input = len(db.files(directory=config.img_input_dir, extension=config.img_input_ext, protocol=config.protocol, **config.all_files_options))
  n_jobs = int(math.ceil(n_input / float(config.N_MAX_FILES_PER_JOB)))
  cmd_gmmstats =  [
                    'gmm_stats.py',  
                    '--config-file=%s' % args.config_file, 
                    '--grid'
                  ]
  if args.nogrid: subprocess.call(cmd_gmmstats)
  else:
    job_gmmstats = submit(jm, cmd_gmmstats, dependencies=[job_ubm.id()], array=(1,n_jobs,1)) 
    print 'submitted:', job_gmmstats

  # Trains the ISV base
  cmd_jfabase = [ 
              'isv_base.py', 
              '--config-file=%s' % args.config_file, 
              '--grid'
            ]
  if args.nogrid: subprocess.call(cmd_jfabase)
  else:
    job_jfabase = submit(jm, cmd_jfabase, dependencies=[job_gmmstats.id()], array=None, queue='q1d', mem='8G')
    print 'submitted:', job_jfabase
 
  # Computes T-Norm models if required
  job_tnorm_L = []
  if config.zt_norm:
    for group in ['dev','eval']:
      n_array_jobs = len(db.Tmodels(protocol=config.protocol, groups=(group,)))
      cmd_tnorm = [
                    'jfa_tmodels.py',
                    '--group=%s' % group,
                    '--config-file=%s' % args.config_file,
                    '--grid'
                  ]
      if args.nogrid: subprocess.call(cmd_tnorm)
      else:
        job_tnorm_int = submit(jm, cmd_tnorm, dependencies=[job_jfabase.id()], array=(1,n_array_jobs,1), queue='q1d', mem='4G')
        job_tnorm_L.append(job_tnorm_int.id())
        print 'submitted:', job_tnorm_int

  # Generates the models 
  job_models_L = []
  for group in ['dev','eval']:
    n_array_jobs = len(db.models(protocol=config.protocol, groups=(group,)))
    cmd_models = [
                  'jfa_models.py',
                  '--group=%s' % group,
                  '--config-file=%s' % args.config_file,
                  '--grid'
                  ]
    if args.nogrid: subprocess.call(cmd_models)
    else:
      job_models_int = submit(jm, cmd_models, dependencies=[job_jfabase.id()], array=(1,n_array_jobs,1), queue='q1d', mem='4G')
      job_models_L.append(job_models_int.id())
      print 'submitted:', job_models_int

  # Compute raw scores (and A matrix for ZT-Norm)
  job_scores_A = []
  for group in ['dev','eval']:
    deps = job_models_L
    deps.append(job_gmmstats.id())
    deps.append(job_jfabase.id())
    n_array_jobs = 0
    model_ids = sorted(db.models(protocol=config.protocol, groups=(group,)))
    for model_id in model_ids:
      n_probes_for_model = len(db.files(groups=group, protocol=config.protocol, purposes='probe', model_ids=(model_id,)))
      n_splits_for_model = int(math.ceil(n_probes_for_model / float(config.N_MAX_PROBES_PER_JOB)))
      n_array_jobs += n_splits_for_model
    cmd_scores_A = [
                  'jfa_scores_A.py',
                  '--group=%s' % group,
                  '--config-file=%s' % args.config_file,
                  '--grid'
                  ]
    if args.nogrid: subprocess.call(cmd_scores_A)
    else:
      job_scores_int = submit(jm, cmd_scores_A, dependencies=deps, array=(1,n_array_jobs,1), queue='q1d', mem='8G')
      job_scores_A.append(job_scores_int.id())
      print 'submitted:', job_scores_int

  # Merges the raw scores
  job_scores_Am = []
  cmd_scores_Am =  [
                    'scores_A_merge.py',  
                    '--config-file=%s' % args.config_file, 
                    '--grid'
                  ]
  if args.nogrid: subprocess.call(cmd_scores_Am)
  else:
    job_scores_Am = submit(jm, cmd_scores_Am, dependencies=job_scores_A) 
    print 'submitted:', job_scores_Am
 
  
  # Computes the B matrix for ZT-Norm
  job_scores_B = []
  if config.zt_norm:
    for group in ['dev','eval']:
      deps = job_models_L
      deps.append(job_gmmstats.id())
      deps.append(job_jfabase.id())
      # Number of models
      n_model_ids = len(db.models(protocol=config.protocol, groups=(group,)))
      # Number of Z-Norm impostor samples
      n_zsamples = len(db.Zfiles(protocol=config.protocol, groups=group))
      n_zsamples_splits = int(math.ceil(n_zsamples / float(config.N_MAX_PROBES_PER_JOB)))
      # Number of array jobs 
      n_array_jobs = n_model_ids * n_zsamples_splits
      cmd_scores_B = [
                    'jfa_scores_B.py',
                    '--group=%s' % group,
                    '--config-file=%s' % args.config_file,
                    '--grid'
                    ]
      if args.nogrid: subprocess.call(cmd_scores_B)
      else:
        job_scores_int = submit(jm, cmd_scores_B, dependencies=deps, array=(1,n_array_jobs,1), queue='q1d', mem='4G')
        job_scores_B.append(job_scores_int.id())
        print 'submitted:', job_scores_int

  # Merges the B matrices 
  job_scores_Bm = []
  cmd_scores_Bm =  [
                    'scores_B_merge.py',  
                    '--config-file=%s' % args.config_file, 
                    '--grid'
                  ]
  if args.nogrid: subprocess.call(cmd_scores_Bm)
  else:
    job_scores_Bm = submit(jm, cmd_scores_Bm, dependencies=job_scores_B) 
    print 'submitted:', job_scores_Bm
 
 
  # Computes the C matrices for ZT-Norm
  job_scores_C = []
  if config.zt_norm:
    for group in ['dev','eval']:
      deps = job_tnorm_L
      deps.append(job_gmmstats.id())
      deps.append(job_jfabase.id())

      n_array_jobs = 0
      # (sorted) list of models
      models_ids = sorted(db.models(protocol=config.protocol, groups=group))
      # Number of T-Norm models
      n_tmodels_ids = len(db.Tmodels(protocol=config.protocol, groups=group))
      n_probes = len(db.files(protocol=config.protocol, purposes='probe', groups=group))
      n_splits = int(math.ceil(n_probes / float(config.N_MAX_PROBES_PER_JOB)))
      n_array_jobs = n_splits * n_tmodels_ids

      cmd_scores_C = [
                    'jfa_scores_C.py',
                    '--group=%s' % group,
                    '--config-file=%s' % args.config_file,
                    '--grid'
                    ]
      if args.nogrid: subprocess.call(cmd_scores_C)
      else:
        job_scores_int = submit(jm, cmd_scores_C, dependencies=deps, array=(1,n_array_jobs,1), queue='q1d', mem='4G')
        job_scores_C.append(job_scores_int.id())
        print 'submitted:', job_scores_int

  # Merges the C matrices 
  job_scores_Cm = []
  cmd_scores_Cm =  [
                    'scores_C_merge.py',  
                    '--config-file=%s' % args.config_file, 
                    '--grid'
                  ]
  if args.nogrid: subprocess.call(cmd_scores_Cm)
  else:
    job_scores_Cm = submit(jm, cmd_scores_Cm, dependencies=job_scores_C) 
    print 'submitted:', job_scores_Cm
 

  # Computes the D matrices for ZT-Norm
  job_scores_D = []
  if config.zt_norm:
    for group in ['dev','eval']:
      deps = job_tnorm_L
      deps.append(job_gmmstats.id())
      deps.append(job_jfabase.id())

      # Number of T-Norm models
      n_tnorm_models_ids = len(db.Tmodels(protocol=config.protocol, groups=group))
      # Number of Z-Norm impostor samples
      n_zsamples = len(db.Zfiles(protocol=config.protocol, groups=group))
      n_zsamples_splits = math.ceil(n_zsamples / float(config.N_MAX_PROBES_PER_JOB))
      # Number of jobs
      n_array_jobs = int(n_zsamples_splits) * n_tnorm_models_ids
      cmd_scores_D = [
                    'jfa_scores_D.py',
                    '--group=%s' % group,
                    '--config-file=%s' % args.config_file,
                    '--grid'
                    ]
      if args.nogrid: subprocess.call(cmd_scores_D)
      else:
        job_scores_int = submit(jm, cmd_scores_D, dependencies=deps, array=(1,n_array_jobs,1), queue='q1d', mem='4G')
        job_scores_D.append(job_scores_int.id())
        print 'submitted:', job_scores_int

  # Merges the D matrices 
  job_scores_Dm = []
  cmd_scores_Dm =  [
                    'scores_D_merge.py',  
                    '--config-file=%s' % args.config_file, 
                    '--grid'
                  ]
  if args.nogrid: subprocess.call(cmd_scores_Dm)
  else:
    job_scores_Dm = submit(jm, cmd_scores_Dm, dependencies=job_scores_D) 
    print 'submitted:', job_scores_Dm

  # Computes the ZT-Norm
  job_scores_ZT = []
  if config.zt_norm:
    cmd_scores_ZT = [ 
              'scores_ztnorm.py', 
              '--config-file=%s' % args.config_file, 
              '--grid'
              ]

  if args.nogrid: subprocess.call(cmd_scores_ZT)
  else:
    job_scores_ZT = submit(jm, cmd_scores_ZT, dependencies=[job_scores_Am.id(), job_scores_Bm.id(),job_scores_Cm.id(), job_scores_Dm.id()], queue='q1d', mem='4G')
    print 'submitted:', job_scores_ZT
 
  # Concatenates the scores
  cmd_cat = [ 
              'concatenate_scores.py', 
              '--config-file=%s' % args.config_file, 
              '--grid'
            ]
  if args.nogrid: subprocess.call(cmd_cat)
  else:
    job_cat = submit(jm, cmd_cat, dependencies=[job_scores_Am.id(),job_scores_ZT.id()], array=None)
    print 'submitted:', job_cat


if __name__ == '__main__':
  main()
