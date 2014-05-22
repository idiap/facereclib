#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <manuel.guenther@idiap.ch>
# Tue Jul 2 14:52:49 CEST 2013
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import print_function

"""
This script parses through the given directory, collects all results of
verification experiments that are stored in file with the given file name.
It supports the split into development and test set of the data, as well as
ZT-normalized scores.

All result files are parsed and evaluated. For each directory, the following
information are given in columns:

  * The Equal Error Rate of the development set
  * The Equal Error Rate of the development set after ZT-Normalization
  * The Half Total Error Rate of the evaluation set
  * The Half Total Error Rate of the evaluation set after ZT-Normalization
  * The sub-directory where the scores can be found

The measure type of the development set can be changed to compute "HTER" or
"FAR" thresholds instead, using the --criterion option.
"""


import sys, os, bob, glob
import argparse
from .. import utils

def command_line_arguments(command_line_parameters):
  """Parse the program options"""

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-d', '--devel-name', dest="dev", default="scores-dev", help = "Name of the file containing the development scores")
  parser.add_argument('-e', '--eval-name', dest="eval", default="scores-eval", help = "Name of the file containing the evaluation scores")
  parser.add_argument('-D', '--directory', default=".", help = "The directory where the results should be collected from; might include search patterns as '*'.")
  parser.add_argument('-n', '--nonorm-dir', dest="nonorm", default="nonorm", help = "Directory where the unnormalized scores are found")
  parser.add_argument('-z', '--ztnorm-dir', dest="ztnorm", default = "ztnorm", help = "Directory where the normalized scores are found")
  parser.add_argument('-s', '--sort', action='store_true', help = "Sort the results")
  parser.add_argument('-k', '--sort-key', dest='key', default = 'nonorm-dev', choices= ('nonorm-dev','nonorm-eval','ztnorm-dev','ztnorm-eval','dir'),
      help = "Sort the results according to the given key")
  parser.add_argument('-c', '--criterion', dest='criterion', default = 'EER', choices = ('EER', 'HTER', 'FAR'),
      help = "Minimize the threshold on the development set according to the given criterion")

  parser.add_argument('-o', '--output', help = "Name of the output file that will contain the EER/HTER scores")
  parser.add_argument('-p', '--parser', default = '4column', choices = ('4column', '5column'), help="The style of the resulting score files")

  parser.add_argument('--self-test', action='store_true', help=argparse.SUPPRESS)

  utils.add_logger_command_line_option(parser)

  # parse arguments
  args = parser.parse_args(command_line_parameters)

  utils.set_verbosity_level(args.verbose)

  # assign the score file parser
  args.parser = {'4column' : bob.measure.load.split_four_column, '5column' : bob.measure.load.split_five_column}[args.parser]

  return args

class Result:
  """Class for collecting the results of one experiment."""
  def __init__(self, dir, args):
    self.dir = dir
    self.m_args = args
    self.nonorm_dev = None
    self.nonorm_eval = None
    self.ztnorm_dev = None
    self.ztnorm_eval = None

  def __calculate__(self, dev_file, eval_file = None):
    """Calculates the EER and HTER or FRR based on the threshold criterion."""
    dev_neg, dev_pos = self.m_args.parser(dev_file)

    # switch which threshold function to use;
    # THIS f***ing piece of code really is what python authors propose:
    threshold = {
      'EER'  : bob.measure.eer_threshold,
      'HTER' : bob.measure.min_hter_threshold,
      'FAR'  : bob.measure.far_threshold
    } [self.m_args.criterion](dev_neg, dev_pos)

    # compute far and frr for the given threshold
    dev_far, dev_frr = bob.measure.farfrr(dev_neg, dev_pos, threshold)
    dev_hter = (dev_far + dev_frr)/2.0

    if eval_file:
      eval_neg, eval_pos = self.m_args.parser(eval_file)
      eval_far, eval_frr = bob.measure.farfrr(eval_neg, eval_pos, threshold)
      eval_hter = (eval_far + eval_frr)/2.0
    else:
      eval_hter = None
      eval_frr = None

    if self.m_args.criterion == 'FAR':
      return (dev_frr, eval_frr)
    else:
      return (dev_hter, eval_hter)

  def nonorm(self, dev_file, eval_file = None):
    self.nonorm_dev, self.nonorm_eval = self.__calculate__(dev_file, eval_file)

  def ztnorm(self, dev_file, eval_file = None):
    self.ztnorm_dev, self.ztnorm_eval = self.__calculate__(dev_file, eval_file)

  def __str__(self):
    str = ""
    for v in [self.nonorm_dev, self.ztnorm_dev, self.nonorm_eval, self.ztnorm_eval]:
      if v:
        val = "% 2.3f%%"%(v*100)
      else:
        val = "None"
      cnt = 16-len(val)
      str += " "*cnt + val
    str += "        %s"%self.dir
    return str[5:]


results = []

def add_results(args, nonorm, ztnorm = None):
  """Adds results of the given nonorm and ztnorm directories."""
  r = Result(os.path.dirname(nonorm).replace(args.directory+"/", ""), args)
  utils.info("Adding results from directory %s" % r.dir)

  # check if the results files are there
  dev_file = os.path.join(nonorm, args.dev)
  eval_file = os.path.join(nonorm, args.eval)
  if os.path.isfile(dev_file):
    if os.path.isfile(eval_file):
      r.nonorm(dev_file, eval_file)
    else:
      r.nonorm(dev_file)

  if ztnorm:
    dev_file = os.path.join(ztnorm, args.dev)
    eval_file = os.path.join(ztnorm, args.eval)
    if os.path.isfile(dev_file):
      if os.path.isfile(eval_file):
        r.ztnorm(dev_file, eval_file)
      else:
        r.ztnorm(dev_file)

  results.append(r)


def recurse(args, path):
  """Recurse the directory structure and collect all results that are stored in the desired file names."""
  dir_list = os.listdir(path)

  # check if the score directories are included in the current path
  if args.nonorm in dir_list:
    if args.ztnorm in dir_list:
      add_results(args, os.path.join(path, args.nonorm), os.path.join(path, args.ztnorm))
    else:
      add_results(args, os.path.join(path, args.nonorm))

  for e in dir_list:
    real_path = os.path.join(path, e)
    if os.path.isdir(real_path):
      recurse(args, real_path)


def table():
  """Generates a table containing all results in a nice format."""
  A = " "*2 + 'dev  nonorm'+ " "*5 + 'dev  ztnorm' + " "*6 + 'eval nonorm' + " "*4 + 'eval ztnorm' + " "*12 + 'directory\n'
  A += "-"*100+"\n"
  for r in results:
    A += str(r) + "\n"
  return A


def main(command_line_parameters = None):
  """Iterates through the desired directory and collects all result files."""
  args = command_line_arguments(command_line_parameters)

  # collect results
  directories = glob.glob(args.directory)
  for directory in directories:
    recurse(args, directory)

  # sort results if desired
  if args.sort:
    import operator
    results.sort(key=operator.attrgetter(args.key.replace('-','_')))

  # print the results
  if args.self_test:
    table()
  elif args.output:
    f = open(args.output, "w")
    f.writelines(table())
    f.close()
  else:
    print (table())
