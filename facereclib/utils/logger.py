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
import bob.core

def set_formatter(logger_name):
  # this formats the logger to print the name of the logger, the time, the type of message and the message itself
  # So, we have to set the formatter to all handlers registered in Bob
  formatter = logging.Formatter("%(name)s@%(asctime)s -- %(levelname)s: %(message)s")
  for handler in logging.getLogger(logger_name).handlers:
    handler.setFormatter(formatter)

set_formatter('bob')
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
  log_level = {
      0: logging.ERROR,
      1: logging.WARNING,
      2: logging.INFO,
      3: logging.DEBUG
    }[level]

  # set this log level to the facereclib logger
  logger.setLevel(log_level)

  # set the same level for the bob logger
  logging.getLogger('bob').setLevel(log_level)


def add_bob_handlers(logger_name, set_format=True, set_log_level=True):
  """Sets the logging handler of 'bob' to the logger with the given name"""
  # add bob's default handlers to the logger with the given name
  requested_logger = logging.getLogger(logger_name)
  for handler in logging.getLogger('bob').handlers:
    requested_logger.addHandler(handler)
  # set the format of the logger to be identical to the one of the facereclib
  if set_format:
    set_formatter(logger_name)
  # set the log level of the facereclib logger also to the desired logger
  if set_log_level:
    requested_logger.setLevel(logger.level)


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

