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
logger = logging.getLogger("bob")

def add_logger_command_line_option(parser):
  """Adds the verbosity command line option to the given parser."""
  parser.add_argument('-v', '--verbose', action = 'count', default = 0,
    help = 'Increase the verbosity level from 0 (only error messages) to 1 (warnings), 2 (log messages), 3 (debug information).')

def set_verbosity_level(level):
  """Sets the log level to 0: Error; 1: Warn; 2: Info; 3: Debug."""
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

