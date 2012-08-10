#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Roy Wallace <roy.wallace@idiap.ch>

from annotations import read_annotations

import os
import bob
import re

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
    


def convertScoreToList(scores, probes):
  ret = []
  i = 0
  for k in sorted(probes.keys()):
    ret.append((probes[k][1], probes[k][2], probes[k][3], probes[k][4], scores[i]))
    i+=1
  return ret


def convertScoreDictToList(scores, probes):
  ret = []
  i = 0
  for k in sorted(probes.keys()):
    ret.append((probes[k][1], probes[k][2], probes[k][3], probes[k][4], scores[i]))
    i+=1
  return ret

def convertScoreListToList(scores, probes):
  ret = []
  i = 0
  for p in probes:
    ret.append((p[1], p[2], p[3], p[4], scores[i]))
    i+=1
  return ret

def probes_used_generate_vector(probe_files_full, probe_files_model):
  """Generates boolean matrices indicating which are the probes for each model"""
  import numpy as np
  C_probesUsed = np.ndarray((len(probe_files_full),), 'bool')
  C_probesUsed.fill(False)
  c=0 
  for k in sorted(probe_files_full.keys()):
    if probe_files_model.has_key(k): C_probesUsed[c] = True
    c+=1
  return C_probesUsed

def probes_used_extract_scores(full_scores, same_probes):
  """Extracts a matrix of scores for a model, given a probes_used row vector of boolean"""
  if full_scores.shape[1] != same_probes.shape[0]: raise "Size mismatch"
  import numpy as np
  model_scores = np.ndarray((full_scores.shape[0],np.sum(same_probes)), 'float64')
  c=0
  for i in range(0,full_scores.shape[1]):
    if same_probes[i]:
      for j in range(0,full_scores.shape[0]):
        model_scores[j,c] = full_scores[j,i]
      c+=1
  return model_scores 

######

class VideoFrameContainer:
  """A class for reading, manipulating and saving video content. 
  A VideoFrameContainer contains data for each of several frames. The data for a frame may represent e.g. a still image, or features extracted from an image. When loaded from or saved to a HDF5 file format, the contents are as follows: 
      /data/<frame_id>, where each <frame_id> is an integer
      /quality/<frame_id> (optional), where each <frame_id> is an integer, stores a vector of quality measures
  """

  def __init__(self, filename = None):
    self._frames = []
    if filename:
      # Read content (frames) from HDF5File
      f = bob.io.HDF5File(filename, "r")
      f.cd('/data/')
      for path in f.paths():

        # Resolve frame_id
        m = re.match('/data/([0-9]*)', path)
        if not m: raise Exception('Failed to read frame_id')
        frame_id = int(m.group(1))

        # Read frame
        data = f.read(path)
        # - read corresponding quality vector if provided
        if f.has_group('/quality') and f.has_key('/quality/' + str(frame_id)):
          quality = f.read('/quality/' + str(frame_id))
        else:
          quality = None

        self._frames.append((frame_id, data, quality))

      del f

  def frames(self):
    """Generator that returns the 3-tuple (frame_id, data, quality) for each frame."""
    for frame in self._frames:
      yield frame

  def add_frame(self,frame_id,frame,quality=None):
    self._frames.append((frame_id,frame,quality))

  def save(self,f):
    """ Save to the specified HDF5File """
    f.create_group('/data')
    f.create_group('/quality')
    for (frame_id, data, quality) in self._frames:
      f.set('/data/' + str(frame_id), data)
      if quality is not None:
        f.set('/quality/' + str(frame_id), quality)

