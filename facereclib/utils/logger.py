#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu Jul 19 17:09:55 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

import logging
import bob

# this formats the logger to print the name of the logger, the time, the type of message and the message itself
# So, we have to set the formatter to all handlers registered in Bob
formatter = logging.Formatter("%(name)s@%(asctime)s -- %(levelname)s: %(message)s")
logger = logging.getLogger("bob")
for handler in logger.handlers:
  handler.setFormatter(formatter)

# this defined our own logger as a child of Bob's logger,
# so that we can distinguish logs of Bob and our own logs
logger = logging.getLogger("bob.facereclib")

def add_logger_command_line_option(parser):
  """Adds the verbosity command line option to the given parser."""
  parser.add_argument('-v', '--verbose', action = 'count', default = 0,
    help = "Increase the verbosity level from 0 (only error messages) to 1 (warnings), 2 (log messages), 3 (debug information) by adding the --verbose option as often as desired (e.g. '-vvv' for debug).")

def set_verbosity_level(level):
  """Sets the log level to 0: Error; 1: Warn; 2: Info; 3: Debug."""
  if level not in range(0,4):
    raise ValueError("The verbosity level %d does not exist. Please reduce the number of '--verbose' parameters in your call" % level)
  # set up the verbosity level of the logging system
  logger.setLevel({
      0: logging.ERROR,
      1: logging.WARNING,
      2: logging.INFO,
      3: logging.DEBUG
    }[level])


def debug(text):
  """Writes the given text to the debug stream."""
  logger.debug(text)

def info(text):
  """Writes the given text to the info stream."""
  logger.info(text)

def warn(text):
  """Writes the given text to the warning stream."""
  logger.warning(text)

def error(text):
  """Writes the given text to the error stream."""
  logger.error(text)

