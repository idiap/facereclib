#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Roy Wallace <roy.wallace@idiap.ch>

from . import histogram
from . import tests
from . import resources
from .logger import add_logger_command_line_option, set_verbosity_level, add_bob_handlers, debug, info, warn, error
from .annotations import read_annotations
from .grid import GridParameters

import os
import bob
import numpy
import tempfile, tarfile

def load(file):
  """Loads data from file. The given file might be an HDF5 file open for reading or a string."""
  if isinstance(file, bob.io.HDF5File):
    return file.read("array")
  else:
    return bob.io.load(file)

def save(data, file, compression=0):
  """Saves the data to file using HDF5. The given file might be an HDF5 file open for writing, or a string.
  If the given data contains a ``save`` method, this method is called with the given HDF5 file.
  Otherwise the data is written to the HDF5 file using the given compression."""
  f = file if isinstance(file, bob.io.HDF5File) else bob.io.HDF5File(file, 'w')
  if hasattr(data, 'save'):
    data.save(f)
  else:
    f.set("array", data, compression=compression)


def load_compressed(filename, compression_type='bz2'):
  """Extracts the data to a temporary HDF5 file using HDF5 and reads its contents.
  Note that, though the file name is .hdf5, it contains compressed data!
  Accepted compression types are 'gz', 'bz2', ''"""
  # create temporary HDF5 file name
  hdf5_file_name = tempfile.mkstemp('.hdf5', 'frl_')[1]

  # create tar file
  tar = tarfile.open(filename, mode="r:"+compression_type)
  memory_file = tar.extractfile(tar.next())
  real_file = open(hdf5_file_name, 'wb')
  real_file.write(memory_file.read())
  del memory_file
  real_file.close()
  tar.close()

  # now, read from HDF5
  hdf5 = bob.io.HDF5File(hdf5_file_name, 'r')
  data = hdf5.read("array")
  del hdf5

  # clean up the mess
  os.remove(hdf5_file_name)

  return data


def save_compressed(data, filename, compression_type='bz2', create_link=False):
  """Saves the data to a temporary file using HDF5.
  Afterwards, the file is compressed using the given compression method and saved using the given file name.
  Note that, though the file name will be .hdf5, it will contain compressed data!
  To be able to read the data using the real tools, a link with the correct extension might is created, when create_link is set to True.
  Accepted compression types are 'gz', 'bz2', ''"""
  # create file in temporary storage
  hdf5_file_name = tempfile.mkstemp('.hdf5', 'frl_')[1]
  hdf5 = bob.io.HDF5File(hdf5_file_name, 'w')
  hdf5.set("array", data)
  # assure that the content is written
  del hdf5
  # create tar file
  tar = tarfile.open(filename, mode="w:"+compression_type)
  tar.add(hdf5_file_name, os.path.basename(filename))
  tar.close()

  # clean up the mess
  os.remove(hdf5_file_name)

  if create_link:
    extension = {'':'.tar', 'bz2':'.tar.bz2', 'gz':'tar.gz'}[compression_type]
    link_file = filename+extension
    if not os.path.exists(link_file):
       os.symlink(os.path.basename(filename), link_file)



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
        'median' : numpy.median,
        None : None
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
  if number_of_desired_items is None or number_of_desired_items >= number_of_total_items or number_of_desired_items < 0:
    return range(number_of_total_items)
  increase = float(number_of_total_items)/float(number_of_desired_items)
  # generate a regular quasi-random index list
  return [int((i +.5)*increase) for i in range(number_of_desired_items)]


def command_line(cmdline):
  """Converts the given options to a string that can be executed on command line."""
  c = ""
  for cmd in cmdline:
    if cmd[0] in '/-':
      c += "%s " % cmd
    else:
      c += "'%s' " % cmd
  return c

