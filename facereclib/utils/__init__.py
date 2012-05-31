#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Roy Wallace <roy.wallace@idiap.ch>

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

def sub_dictionary(input_dict, nb_unit_per_subdict, index):
  """Splits a dictionary into subdictionaries of fixed size
     and returns the index subdictionary"""
  sdict = {}
  count = -1
  start_index=nb_unit_per_subdict*index
  stop_index=start_index+nb_unit_per_subdict-1
  for k in sorted(input_dict.keys()):
    count += 1
    if(count<start_index): continue
    if(count>stop_index):  break
    sdict[k] = input_dict[k]
  return sdict

def split_dictionary(input_dict, nb_unit_per_subdict):
  """Splits a dictionary into a list of subdictionaries"""  
  res = []
  sdict = {}
  to_append = False
  for k in sorted(input_dict.keys()):
    sdict[k] = input_dict[k] 
    to_append = True
    # checks if the subdictionary is full
    if(len(sdict) == nb_unit_per_subdict):
      res.append(sdict)
      to_append = False
      sdict = {}

  if(to_append == True):
    res.append(sdict)
  
  return res 


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
  """
  # TODO later: Implement support for reading/writing accompanying quality vectors per frame, i.e.:
  #    /quality/<frame_id> (optional), where each <frame_id> is an integer

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
        frame_id = m.group(0)

        # Read frame
        data = f.read(path)
        self._frames.append((frame_id, data))
      del f

  def frames(self):
    """Generator that returns the contents of each frame.
    Each item is returned as the 2-tuple (frame_id, data),
    sorted by ascending frame_id."""
    for frame in sorted(self._frames, key=lambda x: x[0]):
      yield frame

  def add_frame(self,frame_id,frame):
    self._frames.append((frame_id,frame))

  def save(self,f):
    """ Save to the specified HDF5File """
    f.create_group('/data')
    f.cd('/data/')
    for frame in self._frames:
      frame_id = frame[0]
      data = frame[1]
      f.set(str(frame_id), data)

