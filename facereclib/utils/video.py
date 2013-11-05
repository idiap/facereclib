#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Roy Wallace <roy.wallace@idiap.ch>

import bob
import re
import numpy

class FrameContainer:
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

  def __eq__(self, other):
    """Equality operator between frame containers."""
    if len(self._frames) != len(other._frames): return False
    for i in range(len(self._frames)):
      if self._frames[i][0] != other._frames[i][0]: return False
      if (numpy.abs(self._frames[i][1] - other._frames[i][1]) > 1e-5).any(): return False
      if (self._frames[i][2] != other._frames[i][2]).any(): return False
    return True

###################################
### Frame selector classes ########

class AllFrameSelector:
  """Selects all of the frames of a video."""

  def __call__(self, frame_container):
    """Yields all frames of the specified VideoFrameContainer,
    sorted by ascending frame_id."""
    for frame in sorted(frame_container.frames(), key=lambda x: x[0]):
      yield frame[1]


class FirstNFrameSelector:
  """Selects the first N frames of a video and discards the rest."""

  def __init__(self, N):
    self._N = N

  def __call__(self, frame_container):
    """Yields the first N frames of the specified VideoFrameContainer. The video must contain at least N frames, otherwise the behaviour is unspecified."""
    sorted_frames = sorted(frame_container.frames(), key=lambda x: x[0])
    for n in range(self._N):
      yield sorted_frames[n][1]


class QualityNFrameSelector:
  """Selects the N frames with highest quality, according to the k'th quality vector field (k>=0)"""

  def __init__(self, N, k):
    self._N = N
    self._k = k

  def __call__(self, frame_container):
    """Yields the N frames with highest value in the k'th field of their corresponding quality vectors (k>=0). The VideoFrameContainer must contain at least N frames, otherwise the behaviour is unspecified."""
    sorted_frames = sorted(frame_container.frames(), key=lambda x: x[2][self._k], reverse=True)
    for n in range(self._N):
      yield sorted_frames[n][1]
