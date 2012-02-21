#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import argparse
import utils
import faceverify as verif
import os

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
  parser.add_argument('-c', '--config-dir', type = str, dest='config_dir', default = '.',
                      help = 'Directory where the automatically generated config files should be written into')

  parser.add_argument('parameters', nargs = argparse.REMAINDER)
  
  args = parser.parse_args()

  # first, invoke the face verification script with the normal configuration
  parameters = args.parameters[1:]
  parameters.extend(['-t', args.tool, '-s', args.default_sub_dir])
  
  verif_args = verif.parse_args(parameters)
  
  job_ids = verif.add_grid_jobs(verif_args)
  external_deps = [job_ids['model_dev_N'], job_ids['model_dev_T'], job_ids['model_eval_N'], job_ids['model_eval_T']]
  
  # now, iterate through the list of replacements and change the keyword in the config file
  index = 0
  for value in args.values:
    # read the config file
    infile = open(args.tool, 'r')
    outfile_name = os.path.join(args.config_dir, args.sub_dirs[index], os.path.basename(args.tool))
    utils.ensure_dir(os.path.dirname(outfile_name))
    outfile = open(outfile_name, 'w')
    
    # iterate through the file    
    for line in infile:
      if line.find(args.keyword) == 0:
        # replace the lines by the new values
        outfile.writelines(args.keyword + " = " + args.values[index])
      else:
        outfile.writelines(line)
    
    # close files
    infile.close()
    outfile.close()
    
    # invoke face verification
    parameters = args.parameters[1:]
    parameters.extend(['-t', outfile_name, '-s', args.sub_dirs[index], 
        '--skip-preprocessing', '--skip-feature-extraction-training', '--skip-feature-extraction', '--skip-projection-training', '--skip-projection', '--skip-enroler-training', '--skip-model-enrolment'])
        
    verif_args = verif.parse_args(parameters)
    verif.add_grid_jobs(verif_args, external_dependencies = external_deps)

if __name__ == "__main__":
  main()  
