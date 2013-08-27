#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Roy Wallace <roy.wallace@idiap.ch>

import video
import histogram
import tests
import resources
from logger import add_logger_command_line_option, set_verbosity_level, add_bob_handlers, debug, info, warn, error
from annotations import read_annotations
from grid import GridParameters

import os
import bob
import numpy

def ensure_dir(dirname):
  """ Creates the directory dirname if it does not already exist,
      taking into account concurrent 'creation' on the grid.
      An exception is thrown if a file (rather than a directory) already
      exists. """
  bob.db.utils.makedirs_safe(dirname)


def score_fusion_strategy(strategy_name = 'avarage'):
  """Returns a function to compute a fusion strategy between different scores."""
  try:
    return {
        'average' : numpy.average,
        'min' : min,
        'max' : max,
        'median' : numpy.median
    }[strategy_name]
  except KeyError:
#    warn("score fusion strategy '%s' is unknown" % strategy_name)
    return None


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


def quasi_random_indices(number_of_total_items, number_of_desired_items = None):
  """Returns a quasi-random list of indices that will contain exactly the number of desired indices (or the number of total items in the list, if this is smaller)."""
  # check if we need to compute a sublist at all
  if number_of_desired_items >= number_of_total_items or number_of_desired_items is None or number_of_desired_items < 0:
    return range(number_of_total_items)
  increase = float(number_of_total_items)/float(number_of_desired_items)
  # generate a regular quasi-random index list
  return [int((i +.5)*increase) for i in range(number_of_desired_items)]



