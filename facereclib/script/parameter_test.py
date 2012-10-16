#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import faceverify, faceverify_gbu, faceverify_lfw
import os, shutil, sys
import argparse
from .. import utils
# from docutils.readers.python.pynodes import parameter


# The different steps of the preprocessing chain.
# Use these keywords to change parameters of the specific part
steps = ['preprocessing', 'features', 'projection', 'enroll', 'scores']

# Parts that could be skipped when the dependecies are on the indexed level
skips = [[''],
         ['--skip-preprocessing'],
         ['--skip-feature-extraction-training', '--skip-feature-extraction'],
         ['--skip-projection-training', '--skip-projection'],
         ['--skip-enroller-training', '--skip-model-enrollment']
        ]

# The keywords to parse the job ids to get the according dependencies right
dkeys  = ['DUMMY', 'preprocessing', 'feature_extraction', 'feature_projection', 'enroll']


def next_level(config, index):
  """Searches the config for the next non-empty level, i.e., where the according keyword is defined in the configuration"""
  for i in range(index, len(steps)):
    if hasattr(config, steps[i]):
      return getattr(config, steps[i])
  return None


def write_config_file(args, infile_name, sub_dir, keyword, value):
  """Copies the given configuration file by replacing the lines including the given keyword with the given value.
     The function returnes the name of the newly generated file."""
  # read the config file
  outfile_name = os.path.join(args.config_dir, sub_dir, os.path.basename(infile_name))
  if infile_name == outfile_name:
    shutil.copy(infile_name, infile_name + '~')
    infile = open(infile_name + '~')
  else:
    infile = open(infile_name, 'r')
  utils.ensure_dir(os.path.dirname(outfile_name))
#  print "\nWriting configuration file '%s'\n"%outfile_name
  outfile = open(outfile_name, 'w')

  replacement_count = 0

  value_skips = len(value.split('\n')) - 1
  skip_lines = 0

  # iterate through the file
  for line in infile:
    if line.find(keyword) == 0:
      # replace the lines by the new values
      outfile.writelines(keyword + " = " + value + "\n")
      replacement_count += 1
      skip_lines = value_skips
    else:
      if skip_lines:
        skip_lines -= 1
      else:
        outfile.writelines(line)

#  if not replacement_count:
#    # add the line when it was not replacable before
#    print "Warning! Could not find the keyword '%s' in the given script '%s'! Adding new line at the end!"%(keyword, infile_name)
#    outfile.writelines("\n" + keyword + " = " + value + "\n")

  # close files
  infile.close()
  outfile.close()


  # return the name of the written config file
  return outfile_name


def directory_parameters(args, dirs):
  """This function generates the faceverify parameters that define the directories, where the data is stored.
     The directories are set such that data is reused whenever possible, but disjoint if needed."""
  parameters = []
  last_dir = '.'
  # add directory parameters
  if dirs['preprocessing'] != '':
    if args.preprocessed_image_dir:
      parameters.extend(['--preprocessed-image-directory', os.path.join(args.preprocessed_image_dir, dirs['preprocessing'], 'preprocessed'), skips[1][0]])
    else:
      parameters.extend(['--preprocessed-image-directory', os.path.join(dirs['preprocessing'], 'preprocessed')])
    last_dir = dirs['preprocessing']
  if dirs['features'] != '':
    parameters.extend(['--features-directory', os.path.join(dirs['features'], 'features')])
    parameters.extend(['--extractor-file', os.path.join(dirs['features'], 'Extractor.hdf5')])
    last_dir = dirs['features']
  if dirs['projection'] != '':
    parameters.extend(['--projected-directory', os.path.join(dirs['projection'], 'projected')])
    parameters.extend(['--projector-file', os.path.join(dirs['projection'], 'Projector.hdf5')])
    last_dir = dirs['projection']
  if dirs['enroll'] != '':
    if args.protocol == 'zt':
      parameters.extend(['--models-directories', os.path.join(dirs['enroll'], 'N-Models'), os.path.join(dirs['enroll'], 'T-Models')])
    elif args.protocol == 'gbu' or args.protocol == 'lfw':
      parameters.extend(['--model-directory', os.path.join(dirs['enroll'], 'models')])
    parameters.extend(['--enroller-file', os.path.join(dirs['enroll'], 'Enroler.hdf5')])
    last_dir = dirs['enroll']
  if dirs['scores'] != '':
    parameters.extend(['--score-sub-dir', dirs['scores']])
    last_dir = dirs['scores']

  # add directory for the submitted db
  dbfile = os.path.join(args.db_dir, last_dir, 'submitted.db')
  utils.ensure_dir(os.path.dirname(dbfile))
  parameters.extend(['--submit-db-file', dbfile])

  return parameters


def get_deps(job_ids, index):
  """Returns the dependencies for the given level from the given job ids"""
  # get the dependencies
  deps = []
  for k in sorted(job_ids.keys()):
    for i in range(index+1):
      if k.find(dkeys[i]) != -1:
        deps.append(job_ids[k])

  print "Dependencies for the job at step '%s' are:"%steps[index], deps
  return deps


def get_skips(index):
  """Returns the skip parameters for the given level"""
  # get the skip parameters
  the_skips = []
  for i in range(1,index+1):
    the_skips.extend(skips[i])
#  print "Skips for step '%s' are: "%steps[index], the_skips
  return the_skips


def execute_dependent_task(args, preprocess_file, feature_file, tool_file, dirs, skips, deps):
  """Executes the face verification task using the given feature and tool configurations, setting dependencies to the given dependent jobs"""
  # invoke face verification with the new configuration, including proper dependencies
  parameters = args.parameters[1:]
  parameters.extend(['-p', preprocess_file, '-f', feature_file, '-t', tool_file])
  parameters.extend(directory_parameters(args, dirs))
  parameters.extend(skips)

  global task_count
  task_count += 1
  global fake_job_id

  # let the face verification script parse the parameters
  if args.protocol == 'zt':
    faceverify_script = faceverify_zt
    face_verify_executable = "faceverify_zt.py"
  elif args.protocol == 'gbu':
    faceverify_script = faceverify_gbu
    face_verify_executable = "faceverify_gbu.py"
  elif args.protocol == 'lfw':
    faceverify_script = faceverify_lfw
    face_verify_executable = "faceverify_lfw.py"


  # write executed call to file
  index = parameters.index('--submit-db-file')

  if args.non_existent_only:
    index2 = parameters.index('--sub-directory')
    path = os.path.realpath(os.path.join(args.non_existent_only, parameters[index2+1], os.path.dirname(parameters[index+1])))
    if os.path.exists(path):
      print "Skipping path '" + path + "' since the results already exist."
      return []

  out_file = os.path.join(os.path.dirname(parameters[index+1]), args.submit_call_file)
  f = open(out_file, 'w')
  f.write(os.path.join(os.path.realpath(os.path.dirname(sys.argv[0])), face_verify_executable) + ' ')
  for p in parameters:
    f.write(p + ' ')
  f.close()

  if args.dry_run:
    job_ids = []
    print "Wrote call to file", out_file, "without executing the file"
  else:
    try:
      verif_args = faceverify_script.parse_args(parameters)

      # execute the face verification
      job_ids = faceverify_script.face_verify(verif_args, external_dependencies = deps, external_fake_job_id = fake_job_id)
    except Exception as e:
      print "\nWARNING: The execution of job '" + out_file + "' was rejected! Reason:"
      print '"', e, '"\n'
      job_ids = []
    global job_count
    job_count += len(job_ids)
    fake_job_id += 100
  return job_ids


def remove_keyword(keyword, config):
  """This function removes the given keyword from the given configuration (and returns a copy of it)"""
  new_config = {}
  for key in config.keys():
    if key != keyword:
      new_config[key] = config[key]
  return new_config


global job_ids
job_ids = {}

def execute_recursively(args, config, index, current_setup, dirs, preprocess_file, feature_file, tool_file, dependency_level):
  if current_setup == None:
    return
  if not len(current_setup):
#    print "\nEntering step", steps[index], "for executing task"
    # try if we need to do another level of recursion
    i = index
    while i < len(steps)-1:
      i += 1
      dirs[steps[i]] = dirs[steps[i-1]]
      # copy directory from previous index
      if hasattr(config, steps[i]):
        execute_recursively(args, config, i, next_level(config, i), dirs, preprocess_file, feature_file, tool_file, dependency_level)
        return

    if i == len(steps)-1:
      # we are at the lowest level, execute jobs
      new_job_ids = execute_dependent_task(args, preprocess_file, feature_file, tool_file, dirs, get_skips(dependency_level), get_deps(job_ids, dependency_level))
#
      print "integrating job ids:", new_job_ids
      print "into old job ids:", job_ids
      job_ids.update(new_job_ids)
      print "The registered job ids are now:", job_ids

  else:
#    print "\nEntering step", steps[index], "for recursive calls"
#    print "executing recursively on step '%s' with dependency step '%s'"%(steps[index], steps[dependency_level])
    # read out the current level of recursion
    keyword = sorted(current_setup.keys())[0]
    replacements = current_setup[keyword]
    remaining_setup = remove_keyword(keyword, current_setup)

#    print remaining_setup

    # The first job is dependent on the given dependency level,
    # while the following jobs are dependent on this level only
    first = True
    dir = dirs[steps[index]]
    # iterate through the replacements
    for sub in replacements.keys():
      if len(replacements) > 1:
        dirs[steps[index]] = os.path.join(dir, sub)
      # replace the current keyword with the current replacement
      new_preprocess_file = write_config_file(args, preprocess_file, os.path.join(dirs[steps[index]], 'preprocessing'), keyword, replacements[sub]) if index < 1 else preprocess_file
      new_feature_file = write_config_file(args, feature_file, os.path.join(dirs[steps[index]], 'feature'), keyword, replacements[sub]) if index < 2 else feature_file
      new_tool_file = write_config_file(args, tool_file, os.path.join(dirs[steps[index]], 'tool'), keyword, replacements[sub])
      execute_recursively(args, config, index, remaining_setup, dirs, new_preprocess_file, new_feature_file, new_tool_file, dependency_level if first else index)
      first = False

#  print "Leaving step", steps[index], "\n"

def execute_parallel(args, config, preprocess_file, feature_file, tool_file):
  job_ids = {}
  dependency_level = 0
  for index in range(len(steps)):
    if hasattr(config, steps[index]):
      setup = getattr(config, steps[index])
      for keyword in sorted(setup.keys()):
        first = True
        for dir, replacement in setup[keyword].iteritems():
          dep_level = dependency_level if first else index
          dirs = {}
          for i in range(index):
            dirs[steps[i]] = '.'
          for i in range(index, len(steps)):
            dirs[steps[i]] = os.path.join(keyword, dir)
          # replace the keyword with the current replacement
          new_preprocess_file = write_config_file(args, preprocess_file, os.path.join(dirs[steps[index]], 'preprocessing'), keyword, replacement)
          new_feature_file = write_config_file(args, feature_file, os.path.join(dirs[steps[index]], 'feature'), keyword, replacement)
          new_tool_file = write_config_file(args, tool_file, os.path.join(dirs[steps[index]], 'tool'), keyword, replacement)

          # execute the job
          new_job_ids = execute_dependent_task(args, preprocess_file, feature_file, tool_file, dirs, get_skips(dep_level), get_deps(job_ids, dep_level))
          if len(new_job_ids):
            first = False

          # generate new dependencies
          print "integrating job ids:", new_job_ids
          print "into old job ids:", job_ids
          job_ids.update(new_job_ids)
          print "The registered job ids are now:", job_ids

        dependency_level = index



def main():
  """Main entry point for the parameter test. Try --help to see the parameters that can be specified."""

  raise NotImplementedError("This function is currently not working. It needs a re-design to work with the latest modifications in the FaceRecLib.")
  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  test_group = parser.add_argument_group('Parameters for the parameter tests')

  test_group.add_argument('-p', '--preprocessing', type = str, required = True, metavar = 'FILE',
      help = 'The preprocessing config file to use')
  test_group.add_argument('-f', '--features', type = str, required = True, metavar = 'FILE',
      help = 'The feature extraction config file to use')
  test_group.add_argument('-t', '--tool', type = str, required = True, metavar = 'FILE',
      help = 'The tool you want to use')
  test_group.add_argument('-P', '--protocol', type = str, choices = ['zt', 'gbu', 'lfw'], default = 'zt',
      help = 'The protocol you want to use')

  test_group.add_argument('-c', '--config-file', type = str, dest = 'config', required = True, metavar = 'FILE',
      help = 'The configuration file explaining what to replace by what')

  test_group.add_argument('-C', '--config-dir', type = str, default = '.',
      help = 'Directory where the automatically generated config files should be written into')

  test_group.add_argument('-S', '--submit-db-dir', type = str, dest='db_dir', default = '.',
      help = 'Directory where the submitted.db files should be written into')

  test_group.add_argument('-F', '--submit-call-file', type=str, default = 'call.txt',
      help = 'Name of the file where to write the executed command into')

  test_group.add_argument('-X', '--preprocessed-image-dir', type=str,
      help = 'Relative directory where the preprocessed images are located; implies --skip-preprocessing')

  test_group.add_argument('-Q', '--non-existent-only', type=str,
      help = 'Only start the experiments that have not been executed successfully (i.e., where the given output directory does not exist yet)')

  test_group.add_argument('-u', '--uncorrelated', action='store_true',
      help = 'Execute the single tests uncorrelated.')

  test_group.add_argument('--dry-run', action='store_true',
      help = 'Only generate call files, but do not execute them')


  # These are the parameters that are forwarded to the face verify script. Use -- to separate the parameter
  verif_group = parser.add_argument_group('Parameters for the face verification script')
  verif_group.add_argument('parameters', nargs = argparse.REMAINDER,
      help = "Parameters directly passed to the face verify script. It should at least include the -d (and the -g) option. Use -- to separate this parameters from the parameters of this script. See 'bin/faceverify_[zt,gbu,lfw].py --help' for a complete list of options.")

  # parse arguments
  args = parser.parse_args()

  # read the configuration file
  import imp
  config = imp.load_source('config', args.config)
  dirs = {}
  for t in steps:
    dirs[t] = '.'

  global task_count
  task_count = 0
  global job_count
  job_count = 0
  # fake job id is used in dry run only
  global fake_job_id
  fake_job_id = 0

  if args.uncorrelated:
    execute_parallel(args, config, args.preprocessing, args.features, args.tool)
  else:
    i = 0
    while i < len(steps):
      # test if the config file for the given step is there
      if hasattr(config, steps[i]):
        execute_recursively(args, config, i, next_level(config,i), dirs, args.preprocessing, args.features, args.tool, 0)
        break
      i += 1

  print "\nDone. The number of executed tasks is:", task_count, "which is split up into", job_count, "single jobs"

if __name__ == "__main__":
  main()
