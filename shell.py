#!/usr/bin/env python

"""Sets up the facereclib toolkit using a bob backend. All arguments passed to this
script are forwarded to the underlying bob shell.py.
"""

import os
import sys
import subprocess

# Choose here the bob release you want to use:
DEFAULT_BOB_DIR = '/idiap/group/torch5spro/nightlies/last'

def find_install_dir():
  """Test to see if I find my own libraries, otherwise, we are probably running
  on the SGE grid using the shell.py script as a launcher. In this case, we
  have to find an alternative. You should set always the job with the -cwd
  option and submit from the root of the facereclib package. As an option, you
  can set the environment variable FACERECLIB_DIR using qsub -v
  FACERECLIB_DIR=<bla/bla>"""

  install_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
  test_path = os.path.join(install_dir, 'lib')

  if not os.path.exists(test_path): #not on CWD, try FACERECLIB_DIR
    if os.environ.has_key('FACERECLIB_DIR') and os.environ['FACERECLIB_DIR']:
      install_dir = os.environ['FACERECLIB_DIR']
      test_path = os.path.join(install_dir, 'lib')

  if not os.path.exists(test_path): #not found, maybe is on the PWD
    if os.environ.has_key('PWD') and os.environ['PWD']:
      install_dir = os.environ['PWD']
      test_path = os.path.join(install_dir, 'lib')

  if not os.path.exists(test_path):
    raise RuntimeError, 'You are running this job using a script outside the facereclib root directory. This is not a problem, as long as you set FACERECLIB_DIR on the environment to point to the root directory of the facereclib package.'

  return os.path.realpath(install_dir)

def find_bob_dir(install_dir):
  """Does the best to find the bob installation directory"""
   
  if os.environ.has_key('BOB_DIR') and os.environ['BOB_DIR'] and \
      os.path.exists(os.environ['BOB_DIR']):
    bob = os.environ['BOB_DIR']
  else:
    bob = DEFAULT_BOB_DIR

  if not os.path.exists(bob): # in the facereclib directory?
    bob = os.path.join(install_dir, 'bob')

  if not os.path.exists(bob): # at the same level?
    bob = os.path.join(install_dir, '..', 'bob')

  if not os.path.exists(bob):
    raise RuntimeError, 'Cannot find a suitable bob installation. The facereclib toolkit requires bob to run. You have 3 options: either set the environment variable BOB_DIR to point to the "bin" directory inside the root of a bob installation, create a link from the facereclib installation directory or from one level up. The link has be called "bob".' 

  return os.path.realpath(bob)

# Locates base facereclib toolkit installation directory
install_dir = find_install_dir()
bob = find_bob_dir(install_dir)

print "Using bob directory located here: %s." % bob

path = 'PATH'
pypath = 'PYTHONPATH'

script_path = os.path.join(install_dir, 'script')
lib_path = os.path.join(install_dir, 'lib')

# Copies the environment
new_environ = dict(os.environ)

if new_environ.has_key(path) and new_environ[path]:
  new_environ[path] = ':'.join([script_path, new_environ[path]])
else:
  new_environ[path] = script_path

if new_environ.has_key(pypath) and new_environ[pypath]:
  new_environ[pypath] = ':'.join([lib_path, new_environ[pypath]])
else:
  new_environ[pypath] = lib_path

# execute the bob shell setup
shell = os.path.join(bob, 'bin', 'shell.py')
arguments = [shell] + sys.argv[1:]

try:
  p = subprocess.Popen(arguments, env=new_environ)
except OSError as e:
  # occurs when the file is not executable or not found
  sys.stderr.write("Error executing '%s': %s (%d)\n" % (' '.join(arguments),
    e.strerror, e.errno))
  sys.exit(e.errno)
  
try:
  p.communicate()
except KeyboardInterrupt: # the user CTRL-C'ed
  import signal
  os.kill(p.pid, signal.SIGTERM)
  sys.exit(signal.SIGTERM)

sys.exit(p.returncode)
