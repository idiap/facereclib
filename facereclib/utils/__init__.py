#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Roy Wallace <roy.wallace@idiap.ch>

from . import histogram
from . import tests
from . import resources
from .logger import add_logger_command_line_option, set_verbosity_level, add_bob_handlers, debug, info, warn, error
from .grid import GridParameters

import bob.io.base
import bob.io.image
import bob.ip.color

import os
import numpy
import tempfile, tarfile

def load(file):
  """Loads data from file. The given file might be an HDF5 file open for reading or a string."""
  if isinstance(file, bob.io.base.HDF5File):
    return file.read("array")
  else:
    return bob.io.base.load(file)

def save(data, file, compression=0):
  """Saves the data to file using HDF5. The given file might be an HDF5 file open for writing, or a string.
  If the given data contains a ``save`` method, this method is called with the given HDF5 file.
  Otherwise the data is written to the HDF5 file using the given compression."""
  f = file if isinstance(file, bob.io.base.HDF5File) else bob.io.base.HDF5File(file, 'w')
  if hasattr(data, 'save'):
    data.save(f)
  else:
    f.set("array", data, compression=compression)


def open_compressed(filename, open_flag = 'r', compression_type='bz2'):
  """Opens a compressed HDF5File with the given opening flags.
  For the 'r' flag, the given compressed file will be extracted to a local space.
  For 'w', an empty HDF5File is created.
  In any case, the opened HDF5File is returned, which needs to be closed using the close_compressed() function.
  """
  # create temporary HDF5 file name
  hdf5_file_name = tempfile.mkstemp('.hdf5', 'frl_')[1]

  if open_flag == 'r':
    # extract the HDF5 file from the given file name into a temporary file name
    tar = tarfile.open(filename, mode="r:"+compression_type)
    memory_file = tar.extractfile(tar.next())
    real_file = open(hdf5_file_name, 'wb')
    real_file.write(memory_file.read())
    del memory_file
    real_file.close()
    tar.close()

  return bob.io.base.HDF5File(hdf5_file_name, open_flag)


def close_compressed(filename, hdf5_file, compression_type='bz2', create_link=False):
  """Closes the compressed hdf5_file that was opened with open_compressed.
  When the file was opened for writing (using the 'w' flag in open_compressed), the created HDF5 file is compressed into the given file name.
  To be able to read the data using the real tools, a link with the correct extension might is created, when create_link is set to True.
  """
  hdf5_file_name = hdf5_file.filename
  is_writable = hdf5_file.writable
  hdf5_file.close()

  if is_writable:
    # create compressed tar file
    tar = tarfile.open(filename, mode="w:"+compression_type)
    tar.add(hdf5_file_name, os.path.basename(filename))
    tar.close()

  if create_link:
    extension = {'':'.tar', 'bz2':'.tar.bz2', 'gz':'tar.gz'}[compression_type]
    link_file = filename+extension
    if not os.path.exists(link_file):
       os.symlink(os.path.basename(filename), link_file)

  # clean up locally generated files
  os.remove(hdf5_file_name)



def load_compressed(filename, compression_type='bz2'):
  """Extracts the data to a temporary HDF5 file using HDF5 and reads its contents.
  Note that, though the file name is .hdf5, it contains compressed data!
  Accepted compression types are 'gz', 'bz2', ''"""
  # read from compressed HDF5
  hdf5 = open_compressed(filename, 'r')
  data = hdf5.read("array")
  close_compressed(filename, hdf5)

  return data


def save_compressed(data, filename, compression_type='bz2', create_link=False):
  """Saves the data to a temporary file using HDF5.
  Afterwards, the file is compressed using the given compression method and saved using the given file name.
  Note that, though the file name will be .hdf5, it will contain compressed data!
  Accepted compression types are 'gz', 'bz2', ''"""
  # write to compressed HDF5 file
  hdf5 = open_compressed(filename, 'w')
  hdf5.set("array", data)
  close_compressed(filename, hdf5)



def ensure_dir(dirname):
  """ Creates the directory dirname if it does not already exist,
      taking into account concurrent 'creation' on the grid.
      An exception is thrown if a file (rather than a directory) already
      exists. """
  bob.io.base.create_directories_safe(dirname)


def score_fusion_strategy(strategy_name = 'avarage'):
  """Returns a function to compute a fusion strategy between different scores.

  Different strategies are employed:

  * ``'average'`` : The averaged score is computed using the :py:func:`numpy.average` function.
  * ``'min'`` : The minimum score is computed using the :py:func:`min` function.
  * ``'max'`` : The maximum score is computed using the :py:func:`max` function.
  * ``'median'`` : The median score is computed using the :py:func:`numpy.median` function.
  * ``None`` is also accepted, in which case ``None`` is returned.
  """
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
    return bob.ip.color.rgb_to_gray(image)
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


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
