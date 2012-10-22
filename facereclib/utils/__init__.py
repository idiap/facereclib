#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Roy Wallace <roy.wallace@idiap.ch>

from annotations import read_annotations
import video
import histogram

from logger import add_logger_command_line_option, set_verbosity_level, debug, info, warn, error

import os
import bob
import imp

def ensure_dir(dirname):
  """ Creates the directory dirname if it does not already exist,
      taking into account concurrent 'creation' on the grid.
      An exception is thrown if a file (rather than a directory) already
      exists. """
  try:
    # Tries to create the directory
    os.makedirs(dirname)
  except OSError:
    # Check that the directory exists
    if os.path.isdir(dirname): pass
    else: raise


def gray_channel(image, channel = 'gray'):
  """Returns the desired channel of the given image. Currently, gray, red, green and blue channels are supported."""
  if image.ndim == 2:
    if channel != 'gray':
      raise ValueError("There is no rule to extract a " + channel + " image from a gray level image!")
    return image

  if channel == 'gray':
    return bob.ip.rgb_to_gray(image)
  if channel == 'red':
    return image[0,:,:]
  if channel == 'green':
    return image[1,:,:]
  if channel == 'blue':
    return image[2,:,:]

  raise ValueError("The image channel " + channel + " is not known or not yet implemented")


def read_config_file(file, keyword = None):
  """Use this function to read the given configuration file.
  If a keyword is specified, only the configuration according to this keyword is returned.
  Otherwise a dictionary of the configurations read from the configuration file is returned."""

  if not os.path.exists(file):
    raise IOError("The given configuration file '%s' could not be found" % file)

  import string
  import random
  tmp_config = "".join(random.sample(string.letters, 10))
  config = imp.load_source(tmp_config, file)

  if not keyword:
    return config

  if not hasattr(config, keyword):
    raise ImportError("The desired keyword '%s' does not exist in your configuration file '%s'." %(keyword, file))

  return eval('config.' + keyword)
