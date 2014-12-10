#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
from __future__ import print_function

from . import faceverify, faceverify_gbu, faceverify_lfw

import argparse, os, sys
import copy # for deep copies of dictionaries
from .. import utils

# the configuration read from config file
global configuration
# the place holder key given on command line
global place_holder_key
# the extracted command line arguments
global args
# the job ids as returned by the call to the faceverify function
global job_ids
# first fake job id (useful for the --dry-run option)
global fake_job_id
fake_job_id = 0
# the number of grid jobs that are executed
global job_count
# the total number of experiments run
global task_count
# the directories, where score files will be generated
global score_directories


# The different steps of the processing chain.
# Use these keywords to change parameters of the specific part
steps = ['preprocessing', 'extraction', 'projection', 'enrollment', 'scoring']



def command_line_options(command_line_parameters):
  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-c', '--configuration-file', required = True,
      help = 'The file containing the information what parameters you want to have tested.')

  parser.add_argument('-k', '--place-holder-key', default = '#',
      help = 'The place holder key that starts the place holders which will be replaced.')

  parser.add_argument('-d', '--database', required = True,
      help = 'The database that you want to execute the experiments on.')

  parser.add_argument('-P', '--protocol',
      help = 'The protocol that you want to use (if not specified, the default protocol for the database is used).')

  parser.add_argument('-b', '--sub-directory', required = True,
      help = 'The sub-directory where the files of the current experiment should be stored. Please specify a directory name with a name describing your experiment.')

  parser.add_argument('-p', '--preprocessing',
      help = "The preprocessing to be used (will overwrite the 'preprocessor' in the configuration file)")

  parser.add_argument('-f', '--features',
      help = "The features to be extracted (will overwrite the 'feature_extractor' in the configuration file)")

  parser.add_argument('-t', '--tool',
      help = "The recognition algorithms to be employed (will overwrite the 'tool' in the configuration file)")

  parser.add_argument('-g', '--grid',
      help = 'The SGE grid configuration')

  parser.add_argument('-x', '--executable',
      help = '(optional) The executable to be executed instead of facereclib/script/faceverify.py (taken *always* from the facereclib, not from the bin directory)')

  parser.add_argument('-R', '--result-directory', default = "results",
      help = 'The directory where to write the resulting score files to.')

  parser.add_argument('-T', '--temp-directory', default = "temp",
      help = 'The directory where to write temporary files into.')

  parser.add_argument('-i', '--preprocessed-data-directory',
      help = '(optional) The directory where to read the already preprocessed data from (no preprocessing is performed in this case).')

  parser.add_argument('-s', '--grid-database-directory', default = 'grid_db',
      help = 'Directory where the submitted.sql3 files should be written into (will create sub-directories on need)')

  parser.add_argument('-w', '--write-commands',
      help = '(optional) The file name where to write the calls into (will not write the dependencies, though)')

  parser.add_argument('-q', '--dry-run', action='store_true',
      help = 'Just write the commands to console and mimic dependencies, but do not execute the commands')

  parser.add_argument('-j', '--skip-when-existent', action='store_true',
      help = 'Skip the submission/execution of jobs when the result directory already exists')

  parser.add_argument('-N', '--replace-variable',
      help = 'Use the given variable instead of the "replace" keyword in the configuration file')

  parser.add_argument('parameters', nargs = argparse.REMAINDER,
      help = "Parameters directly passed to the face verify script. Use -- to separate this parameters from the parameters of this script. See 'bin/faceverify.py --help' for a complete list of options.")

  utils.add_logger_command_line_option(parser)

  global args
  args = parser.parse_args(command_line_parameters)
  utils.set_verbosity_level(args.verbose)

  if args.executable:
    global faceverify
    faceverify = __import__('importlib').import_module(args.executable)




def extract_values(replacements, indices):
  """Extracts the value dictionary from the given dictionary of replacements"""
  extracted_values = {}
  for place_holder in replacements.keys():
    # get all occurrences of the place holder key
    parts = place_holder.split(place_holder_key)
    # only one part -> no place holder key found -> no strings to be extracted
    if len(parts) == 1:
      continue

    keys = [part[:1] for part in parts[1:]]

    value_index = indices[place_holder]

    entries = replacements[place_holder]
    entry_key = sorted(entries.keys())[value_index]

    # check that the keys are unique
    for key in keys:
      if key in extracted_values:
        raise ValueError("The replacement key '%s' was defined multiple times. Please use each key only once."%key)

    # extract values
    if len(keys) == 1:
      extracted_values[keys[0]] = entries[entry_key]

    else:
      for i in range(len(keys)):
        extracted_values[keys[i]] = entries[entry_key][i]

  return extracted_values


def replace(string, replacements):
  """Replaces the place holders in the given string with the according values from the values dictionary."""
  # get all occurrences of the place holder key
  parts = string.split(place_holder_key)
  # only one part -> no place holder key found -> return the whole string
  if len(parts) == 1:
    return string

  keys = [part[:1] for part in parts[1:]]

  retval = parts[0]
  for i in range(0, len(keys)):
    # replace the place holder by the desired string and add the remaining of the command
    retval += str(replacements[keys[i]]) + str(parts[i+1][1:])

  return retval


def create_command_line(replacements):
  """Creates the parameters for the function call that will be given to the faceverify script."""
  # get the values to be replaced with
  values = {}
  for key in configuration.replace:
    values.update(extract_values(configuration.replace[key], replacements))
  # replace the place holders with the values
  call = [sys.argv[0], '--database', args.database]
  if args.protocol:
    call += ['--protocol', args.protocol]
  call += ['--temp-directory', args.temp_directory, '--result-directory', args.result_directory]
  return call + [
      '--preprocessing', replace(configuration.preprocessor, values),
      '--features', replace(configuration.feature_extractor, values),
      '--tool', replace(configuration.tool, values),
      '--imports'
  ] + configuration.imports



# Parts that could be skipped when the dependecies are on the indexed level
skips = [[''],
         ['--skip-preprocessing'],
         ['--skip-extractor-training', '--skip-extraction'],
         ['--skip-projector-training', '--skip-projection'],
         ['--skip-enroller-training', '--skip-enrollment']
        ]

# The keywords to parse the job ids to get the according dependencies right
dependency_keys  = ['DUMMY', 'preprocess', 'extract', 'project', 'enroll']


def directory_parameters(directories):
  """This function generates the faceverify parameters that define the directories, where the data is stored.
  The directories are set such that data is reused whenever possible, but disjoint if needed."""
  def join_dirs(index, subdir):
    # collect sub-directories
    dirs = []
    for i in range(index+1):
      dirs.extend(directories[steps[i]])
    if not dirs:
      return subdir
    else:
      dir = dirs[0]
      for d in dirs[1:]:
        dir = os.path.join(dir, d)
      return os.path.join(dir, subdir)

  global args
  parameters = []

  # add directory parameters
  # - preprocessing
  if args.preprocessed_data_directory:
    parameters.extend(['--preprocessed-data-directory', os.path.join(args.preprocessed_data_directory, join_dirs(0, 'preprocessed'))] + skips[1])
  else:
    parameters.extend(['--preprocessed-data-directory', join_dirs(0, 'preprocessed')])

  # - feature extraction
  parameters.extend(['--features-directory', join_dirs(1, 'features')])
  parameters.extend(['--extractor-file', join_dirs(1, 'Extractor.hdf5')])

  # - feature projection
  parameters.extend(['--projected-features-directory', join_dirs(2, 'projected')])
  parameters.extend(['--projector-file', join_dirs(2, 'Projector.hdf5')])

  # - model enrollment
  # TODO: other parameters for other scripts?
  parameters.extend(['--models-directories', join_dirs(3, 'N-Models'), join_dirs(3, 'T-Models')])
  parameters.extend(['--enroller-file', join_dirs(3, 'Enroller.hdf5')])

  # - scoring
  parameters.extend(['--score-sub-directory', join_dirs(4, 'scores')])

  parameters.extend(['--sub-directory', args.sub_directory])

  global score_directories
  score_directories.append(join_dirs(4, 'scores'))

  # grid database
  if args.grid:
    # we get one database per preprocessing job (all others might have job inter-dependencies)
    parameters.extend(['--submit-db-file', os.path.join(args.grid_database_directory, join_dirs(0, 'submitted.sql3'))])

  return parameters


def check_requirements(replacements):
  # check if the requirement are met
  global configuration
  values = {}
  for key in configuration.replace:
    values.update(extract_values(configuration.replace[key], replacements))
  for requirement in configuration.requirements:
    test = replace(requirement, values)
    while not isinstance(test, bool):
      test = eval(test)
    if not test:
      return False
  return True


def execute_dependent_task(command_line, directories, dependency_level):
  # add other command line arguments
  if args.grid:
    command_line.extend(['--grid', args.grid])
  if args.verbose:
    command_line.append('-' + 'v'*args.verbose)

  # create directory parameters
  command_line.extend(directory_parameters(directories))

  # add skip parameters according to the dependency level
  for i in range(1, dependency_level+1):
    command_line.extend(skips[i])

  if args.parameters is not None:
    command_line.extend(args.parameters[1:])

  # write the command to file?
  if args.write_commands:
    index = command_line.index('--submit-db-file')
    command_file = os.path.join(os.path.dirname(command_line[index+1]), args.write_commands)
    with open(command_file, 'w') as f:
      f.write('bin/faceverify.py ')
      for p in command_line[1:]:
        f.write(p + ' ')
      f.close()
    utils.info("Wrote command line into file '%s'" % command_file)

  # extract dependencies
  global job_ids
  dependencies = []
  for k in sorted(job_ids.keys()):
    for i in range(1, dependency_level+1):
      if k.find(dependency_keys[i]) != -1:
        dependencies.append(job_ids[k])

  # execute the command
  new_job_ids = {}
  try:
    verif_args = faceverify.parse_args(command_line[1:])
    result_dir = os.path.join(verif_args.user_directory, verif_args.sub_directory, verif_args.score_sub_directory)
    if not args.skip_when_existent or not os.path.exists(result_dir):
      # get the command line parameter for the result directory
      if args.dry_run:
        if args.verbose:
          print ("Would have executed job", utils.command_line(command_line), "with dependencies", dependencies)
      else:
        # execute the face verification experiment
        global fake_job_id
        new_job_ids = faceverify.face_verify(verif_args, command_line, external_dependencies = dependencies, external_fake_job_id = fake_job_id)
    else:
      utils.info("Skipping execution of %s since result directory '%s' already exists" % (utils.command_line(command_line), result_dir))

  except Exception as e:
    utils.error("The execution of job was rejected!\n%s\n Reason:\n%s"%(utils.command_line(command_line), e))

  # some statistics
  global job_count, task_count
  job_count += len(new_job_ids)
  task_count += 1
  fake_job_id += 100
  job_ids.update(new_job_ids)


def create_recursive(replace_dict, step_index, directories, dependency_level, keys=[]):
  """Iterates through all the keywords and replaces all place holders with all keywords in a defined order."""

  # check if we are at the lowest level
  if step_index == len(steps):
    # create a call and execute it
    if check_requirements(replace_dict):
      execute_dependent_task(create_command_line(replace_dict), directories, dependency_level)
  else:
    if steps[step_index] not in directories:
      directories[steps[step_index]] = []

    # we are at another level
    if steps[step_index] not in configuration.replace.keys():
      # nothing to be replaced here, so just go to the next level
      create_recursive(replace_dict, step_index+1, directories, dependency_level)
    else:
      # iterate through the keys
      if keys == []:
        # call this function recursively by defining the set of keys that we need
        create_recursive(replace_dict, step_index, directories, dependency_level, keys = sorted(configuration.replace[steps[step_index]].keys()))
      else:
        # create a deep copy of the replacement dict to be able to modify it
        replace_dict_copy = copy.deepcopy(replace_dict)
        directories_copy = copy.deepcopy(directories)
        # iterate over all replacements for the first of the keys
        key = keys[0]
        replacement_directories = sorted(configuration.replace[steps[step_index]][key])
        directories_copy[steps[step_index]].append("")
        new_dependency_level = dependency_level
        for replacement_index in range(len(replacement_directories)):
          # increase the counter of the current replacement
          replace_dict_copy[key] = replacement_index
          directories_copy[steps[step_index]][-1] = replacement_directories[replacement_index]
          # call the function recursively
          if len(keys) == 1:
            # we have to go to the next level
            create_recursive(replace_dict_copy, step_index+1, directories_copy, new_dependency_level)
          else:
            # we have to subtract the keys
            create_recursive(replace_dict_copy, step_index, directories_copy, new_dependency_level, keys = keys[1:])
          new_dependency_level = step_index


def main(command_line_parameters = sys.argv):
  """Main entry point for the parameter test. Try --help to see the parameters that can be specified."""

  global task_count, job_count, job_ids, score_directories
  job_count = 0
  task_count = 0
  job_ids = {}
  score_directories = []

  command_line_options(command_line_parameters[1:])

  global configuration, place_holder_key
  configuration = utils.resources.read_config_file(args.configuration_file)
  place_holder_key = args.place_holder_key

  if args.preprocessing:
    configuration.preprocessor = args.preprocessing
  if args.features:
    configuration.feature_extractor = args.features
  if args.tool:
    configuration.tool = args.tool

  if args.replace_variable is not None:
    exec("configuration.replace = configuration.%s" % args.replace_variable)

  for attribute in ('preprocessor', 'feature_extractor', 'tool'):
    if not hasattr(configuration, attribute):
      raise ValueError("The given configuration file '%s' does not contain the required attribute '%s', and it was not given on command line either" %(args.configuration_file, attribute))

  # extract the dictionary of replacements from the configuration
  if not hasattr(configuration, 'replace'):
    raise ValueError("Please define a set of replacements using the 'replace' keyword.")
  if not hasattr(configuration, 'imports'):
    configuration.imports = ['facereclib']
    utils.info("No 'imports' specified in configuration file '%s' -> using default %s" %(args.configuration_file, configuration.imports))

  if not hasattr(configuration, 'requirements'):
    configuration.requirements = []

  replace_dict = {}
  for step, replacements in configuration.replace.items():
    for key in replacements.keys():
      if key in replace_dict:
        raise ValueError("The replacement key '%s' was defined multiple times. Please use each key only once.")
      # we always start with index 0.
      replace_dict[key] = 0

  # now, iterate through the list of replacements and create the according calls
  create_recursive(replace_dict, step_index = 0, directories = {}, dependency_level = 0)

  # finally, write some information about the
  utils.info("The number of executed tasks is: %d, which are split up into %d jobs that are executed in the grid" %(task_count, job_count))

  return score_directories


if __name__ == "__main__":
  main()
