#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import faceverify as verif
import os
import argparse
# import utils AFTER faceverify since this has set the python path correctly
import utils

def write_config_file(args, infile_name, sub_dir, keyword, value):
  # read the config file
  infile = open(infile_name, 'r')
  outfile_name = os.path.join(args.config_dir, sub_dir, os.path.basename(args.tool))
  utils.ensure_dir(os.path.dirname(outfile_name))
  print "\n\nWriting configuration file '%s'\n\n"%outfile_name
  outfile = open(outfile_name, 'w')
  
  # iterate through the file    
  for line in infile:
    if line.find(keyword) == 0:
      # replace the lines by the new values
      outfile.writelines(keyword + " = " + value)
    else:
      outfile.writelines(line)
  
  # close files
  infile.close()
  outfile.close()
  
  # return the name of the written config file
  return outfile_name
  

def execute_main_task(args, sub_dir):
  # invoke the face verification script with the normal configuration
  parameters = args.parameters[1:]
  parameters.extend(['-t', args.tool, '-s', sub_dir, '--models-directories', os.path.join(sub_dir, 'N-models'), os.path.join(sub_dir, 'T-models')])
  
  verif_args = verif.parse_args(parameters)
  
#  print "Executing main task in sub_dir '%s'"%sub_dir
#  return []
  
  job_ids = verif.add_grid_jobs(verif_args)
  # get the dependencies
  deps = []
  for k in sorted(job_ids.keys()):
    if k.find(args.dep_key) != -1:
      deps.append(job_ids[k])

  return deps

def skips(args):
  skips = ['--skip-preprocessing', '--skip-feature-extraction-training', '--skip-feature-extraction']
  if args.dep_key != 'feature_extraction':
    skips.extend(['--skip-projection-training', '--skip-projection'])
    if args.dep_key != 'feature_projection':
      skips.extend(['--skip-enroler-training', '--skip-model-enrolment'])
    
  return skips

def execute_dependent_task(args, config_file, sub_dir, deps):
  # invoke face verification with the new configuration, including proper dependencies
  parameters = args.parameters[1:]
  parameters.extend(['-t', config_file, '-s', sub_dir, '--models-directories', os.path.join(sub_dir, 'N-models'), os.path.join(sub_dir, 'T-models')])
  parameters.extend(skips(args))
  
  print parameters
  verif_args = verif.parse_args(parameters)
  
#  print "Executing subtask with config file '%s' in sub_dir '%s'"%(config_file,sub_dir)
#  return
  
  verif.add_grid_jobs(verif_args, external_dependencies = deps)



def remove_keyword(keyword, config):
  new_config = {}
  for key in config.keys():
    if key != keyword:
      new_config[key] = config[key]
  return new_config


def default_dir(config, sub_dir = None):
  dir = sub_dir
  for keyword in config.keys():
    for sub_dir in config[keyword]:
      if config[keyword][sub_dir] == None:
        if dir == None:
          dir = sub_dir
        else:
          dir = os.path.join(dir, sub_dir)
  return dir
    

def execute_recursively(args, config, dir, config_file, deps, is_default = True):
  if len(config) == 0:
    # we reached the bottom, so now we can throw a job
    if not is_default:
      execute_dependent_task(args, config_file, dir, deps)
  else:
    # get the first keyword 
    keyword = config.keys()[0]
    replacements = config[keyword]
    remaining_config = remove_keyword(keyword, config)
    
    # iterate through the replacements
    for sub in replacements.keys():
      sub_dir = os.path.join(dir, sub)
      if replacements[sub] == None:
        execute_recursively(args, remaining_config, sub_dir, config_file, deps, is_default)
      else:
        # replace the current keyword with the current replacement
        new_sub_dir = default_dir(remaining_config, sub_dir)
        new_tool = write_config_file(args, config_file, new_sub_dir, keyword, replacements[sub])
        execute_recursively(args, remaining_config, sub_dir, new_tool, deps, False)
    

def main2():
  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('-r', '--tool-config', type = str, dest = 'tool', required = True, metavar = 'FILE', 
                      help = 'The tool you want to use')
  
  parser.add_argument('-c', '--config-file', type = str, dest = 'config', required = True, metavar = 'FILE',
                      help = 'The configuration file explaining what to replace by what')

  parser.add_argument('-C', '--config-dir', type = str, dest='config_dir', default = '.',
                      help = 'Directory where the automatically generated config files should be written into')
  
  parser.add_argument('-d', '--dependency-keyword', type = str, dest = 'dep_key', metavar='KEYWORD',
                      choices = ['feature_extraction', 'feature_projection', 'enrol'], default = 'feature_extraction', 
                      help = 'The keyword defining which kind of dependency to the default job is required')

  parser.add_argument('parameters', nargs = argparse.REMAINDER)
  
  args = parser.parse_args()

  # read the configuration file
  import imp
  config = imp.load_source('config', args.config).test

  # execute face verification with the default setup 
  deps = execute_main_task(args, default_dir(config))
  
  execute_recursively(args, config, ".", args.tool, deps)

  return

  # get the first keyword 
  keyword = config.keys()[0]
  replacements = config[keyword]
  remaining_config = remove_keyword(keyword, config)
  
  # iterate through the replacements
  for sub_dir in replacements.keys():
    if replacements[sub_dir] == None:
      execute_recursively(args, remaining_config, sub_dir, args.tool, deps, True)
    else:
      # replace the current keyword with the current replacement
      new_sub_dir = default_dir(remaining_config, sub_dir)
      new_tool = write_config_file(args, args.tool, new_sub_dir, keyword, replacements[sub_dir])
      execute_recursively(args, remaining_config, sub_dir, new_tool, deps, False)
      
def main():
  """This is the main entry point for computing face verification experiments.
  You just have to specify configuration scripts for any of the steps of the toolchain, which are:
  -- the database
  -- feature extraction (including image preprocessing)
  -- the score computation tool
  -- and the grid configuration (in case, the function should be executed in the grid).
  Additionally, you can skip parts of the toolchain by selecting proper --skip-... parameters.
  If your probe files are not too big, you can also specify the --preload-probes switch to speed up the score computation.
  If files should be re-generated, please specify the --force option (might be combined with the --skip-... options)"""
  
  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('-r', '--tool-config', type = str, dest = 'tool', required = True, metavar = 'FILE', 
                      help = 'The tool you want to use')
                      
  parser.add_argument('-k', '--keyword', metavar = 'STRING', type = str, 
                      help = 'the keyword for the function that should be filled in')
  parser.add_argument('-v', '--values', metavar = 'STRING', type = str, nargs='+',
                      help = 'the values that should be written to the keyword')
  parser.add_argument('-q', '--sub-dirs', metavar = 'STRING', type = str, nargs='+', dest = 'sub_dirs',
                      help = 'the subdirectories to create for each of the testet values')
  parser.add_argument('-Q', '--default-sub-dir', metavar = 'STRING', type = str, default = 'original', dest='default_sub_dir',
                      help = 'the subdirectories to create for each of the testet values')
  parser.add_argument('-C', '--config-dir', type = str, dest='config_dir', default = '.',
                      help = 'Directory where the automatically generated config files should be written into')

  parser.add_argument('parameters', nargs = argparse.REMAINDER)
  
  args = parser.parse_args()
  
  # first, execute the job with the default setup
  deps = execute_main_task(args, args.default_sub_dir)

  # now, iterate through the list of replacements and change the keyword in the config file
  index = 0
  for value in args.values:
    # write config file
    write_config_file(args, args.tool, args.sub_dirs[index], args.keyword, args.values[index])
    
    # invoke face verification
    execute_dependent_task(args, outfile_name, args.sub_dirs[index], deps)
    index += 1

if __name__ == "__main__":
  main2()
