#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import faceverify
import os
import argparse
from .. import utils
# from docutils.readers.python.pynodes import parameter


# The different steps of the preprocessing chain.
# Use these keywords to change parameters of the specific part
steps = ['preprocessing', 'features', 'projection', 'enrol', 'scores']

# Parts that could be skipped when the dependecies are on the indexed level
skips = [[''], 
         ['--skip-preprocessing'], 
         ['--skip-feature-extraction-training', '--skip-feature-extraction'],
         ['--skip-projection-training', '--skip-projection'],
         ['--skip-enroler-training', '--skip-model-enrolment']
        ]

# The keywords to parse the job ids to get the according dependencies right 
dkeys  = ['DUMMY', 'preprocessing', 'feature_extraction', 'feature_projection', 'enrol']


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
  infile = open(infile_name, 'r')
  outfile_name = os.path.join(args.config_dir, sub_dir, os.path.basename(infile_name))
  utils.ensure_dir(os.path.dirname(outfile_name))
#  print "\nWriting configuration file '%s'\n"%outfile_name
  outfile = open(outfile_name, 'w')
  
  replacement_count = 0
  
  # iterate through the file    
  for line in infile:
    if line.find(keyword) == 0:
      # replace the lines by the new values
      outfile.writelines(keyword + " = " + value + "\n")
      replacement_count += 1
    else:
      outfile.writelines(line)
  
  # close files
  infile.close()
  outfile.close()
  
  if not replacement_count:
    raise "Could not find the keyword '%s' in the given script '%s'!"%(keyword, infile_name)
  
  # return the name of the written config file
  return outfile_name


def directory_parameters(args, dirs):
  """This function generates the faceverify parameters that define the directories, where the data is stored. 
     The directories are set such that data is reused whenever possible, but disjoined if needed."""
  parameters = []
  last_dir = '.'  
  # add directory parameters
  if dirs['preprocessing'] != '':
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
  if dirs['enrol'] != '':
    parameters.extend(['--models-directories', os.path.join(dirs['enrol'], 'N-Models'), os.path.join(dirs['enrol'], 'T-Models')]) 
    parameters.extend(['--enroler-file', os.path.join(dirs['enrol'], 'Enroler.hdf5')]) 
    last_dir = dirs['enrol'] 
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


def execute_dependent_task(args, feature_file, tool_file, dirs, skips, deps):
  """Executes the face verification task using the given feature and tool configurations, setting dependencies to the given dependent jobs"""
  # invoke face verification with the new configuration, including proper dependencies
  parameters = args.parameters[1:]
  parameters.extend(['-p', feature_file, '-t', tool_file])
  parameters.extend(directory_parameters(args, dirs))
  parameters.extend(skips)

  global task_count
  task_count += 1

  if args.dry_run:
    print parameters
    return []
  else:
    # let the face verification script parse the parameters
    verif_args = verif.parse_args(parameters)

    # execute the face verification  
    job_ids = verif.add_grid_jobs(verif_args, external_dependencies = deps)
    global job_count
    job_count += len(job_ids) 
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

def execute_recursively(args, config, index, current_setup, dirs, feature_file, tool_file, dependency_level):
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
        execute_recursively(args, config, i, next_level(config, i), dirs, feature_file, tool_file, dependency_level)
        return
      
    if i == len(steps)-1:
      # we are at the lowest level, execute jobs
      new_job_ids = execute_dependent_task(args, feature_file, tool_file, dirs, get_skips(dependency_level), get_deps(job_ids, dependency_level))
#
#      print "integrating job ids:", new_job_ids
#      print "into old job ids:", job_ids
      job_ids.update(new_job_ids)
      print "The registered job ids are now:", job_ids
        
  else:
#    print "\nEntering step", steps[index], "for recursive calls"
#    print "executing recursively on step '%s' with dependency step '%s'"%(steps[index], steps[dependency_level])
    # read out the current level of recursion
    keyword = current_setup.keys()[0]
    replacements = current_setup[keyword]
    remaining_setup = remove_keyword(keyword, current_setup)
    
#    print remaining_setup
    
    # The first job is dependent on the given dependency level, 
    # while the following jobs are dependent on this level only 
    first = True
    dir = dirs[steps[index]]
    # iterate through the replacements
    for sub in replacements.keys():
      dirs[steps[index]] = os.path.join(dir, sub)
      # replace the current keyword with the current replacement
      new_feature_file = write_config_file(args, feature_file, dirs[steps[index]], keyword, replacements[sub]) if index < 2 else feature_file
      new_tool_file = write_config_file(args, tool_file, dirs[steps[index]], keyword, replacements[sub]) if index >= 2 else tool_file
      execute_recursively(args, config, index, remaining_setup, dirs, new_feature_file, new_tool_file, dependency_level if first else index)
      first = False
    
#  print "Leaving step", steps[index], "\n"

def main():
  """Main entry point for the parameter test. Try --help to see the parameters that can be specified."""
  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('-f', '--features', type = str, dest = 'features', required = True, metavar = 'FILE',
                      help = 'The feature extraction config file to use')
  parser.add_argument('-t', '--tool', type = str, dest = 'tool', required = True, metavar = 'FILE', 
                      help = 'The tool you want to use')
  
  parser.add_argument('-c', '--config-file', type = str, dest = 'config', required = True, metavar = 'FILE',
                      help = 'The configuration file explaining what to replace by what')

  parser.add_argument('-C', '--config-dir', type = str, dest='config_dir', default = '.',
                      help = 'Directory where the automatically generated config files should be written into')
  
  parser.add_argument('-S', '--submit-db-dir', type = str, dest='db_dir', default = '.',
                      help = 'Directory where the submitted.db files should be written into')
  
  parser.add_argument('-L', '--log-dir', type = str, dest='log_dir', default = '.',
                      help = 'Directory where the log files should be written into')
  
  parser.add_argument('-d', '--dry-run', action = 'store_true', dest='dry_run', 
                      help = 'Just show the commands and count them, but do not execute them')
  
  # These are the parameters that are forwarded to the face verify script. Use -- to separate the para 
  parser.add_argument('parameters', nargs = argparse.REMAINDER,
                      help = 'Parameters directly passed to the face verify script. It should at least include the -d (and the -g) option. Use -- to separate this parameters from the parameters of this script.')
  
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
  
  i = 0
  while i < len(steps)-1:
    # test if the config file for the given step is there
    if hasattr(config, steps[i]):
      execute_recursively(args, config, i, next_level(config,i), dirs, args.features, args.tool, 0)
      break
    i += 1

  print "\nDone. The number of executed tasks is:", task_count, "which is split up into", job_count, "single jobs"

if __name__ == "__main__":
  main()
