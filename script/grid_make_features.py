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

def submit(job_manager, command, dependencies=[], array=None):
  """Submits one job using our specialized shell wrapper. We hard-code certain
  parameters we like to use. You can change general submission parameters
  directly at this method."""
 
  from gridtk.tools import make_python_wrapper, random_logdir
  name = os.path.splitext(os.path.basename(command[0]))[0]
  logdir = os.path.join('logs', random_logdir())

  use_cmd = make_python_wrapper(FACERECLIB_WRAPPER, command)
  return job_manager.submit(use_cmd, deps=dependencies, cwd=True,
      queue='all.q', stdout=logdir, stderr=logdir, name=name, array=array,
      env=FACERECLIB_WRAPPER_ENVIRONMENT)

def generic_submit(script_filename, config_filename, TOTAL_ARRAY_JOBS, job_manager):
  """Submit the given script on the grid using with the provided configuration file

  Keyword parameters

  script_filename
    The filename of the script to be launched

  config_filename
    The configuration file for the script

  TOTAL_ARRAY_JOBS
    The total number of array jobs that should be launched

  job_manager
    This is the gridtk.manager.JobManager to use for submitting the jobs.

  Returns the gridtk.manager.Job's for all jobs in a python list
  """

  cmd = [
          script_filename,
          '--config-file=%s' % config_filename,
          '--grid'
        ]
  print cmd
  array = (1,TOTAL_ARRAY_JOBS,1)
  job = submit(job_manager, cmd, array=array)
  print 'submitted:', job

  return job


def main():
  """The main entry point, control here the jobs options and other details"""

  # Parses options
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-s', '--script-file', metavar='FILE', type=str,
      dest='script_file', default="", help='Filename of the script to run on the grid (defaults to "%(default)s")')
  parser.add_argument('-c', '--config-file', metavar='FILE', type=str,
      dest='config_file', default="", help='Filename of the configuration file to use to run the script on the grid (defaults to "%(default)s")')
  args = parser.parse_args()

  # Loads the configuration 
  import imp
  config = imp.load_source('config', args.config_file)
  img_input = config.db.files(directory=config.img_input_dir, extension=config.img_input_ext, protocol=config.protocol, **config.all_files_options)
  n_jobs = int(math.ceil(len(img_input) / float(config.N_MAX_FILES_PER_JOB)))

  # Let's create the job manager
  from gridtk.manager import JobManager
  jm = JobManager()

  # Runs the jobs
  jobs = generic_submit(args.script_file, args.config_file, n_jobs, jm)

if __name__ == '__main__':
  main()
