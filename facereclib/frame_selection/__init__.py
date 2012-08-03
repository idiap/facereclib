#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

"""A frame selector filters a subset of the frames from an input VideoFrameContainer"""

import numpy
import bob

# Note: each frame in VideoFrameContainer.frames() is a 3-tuple:
#       (frame_id, data, quality)

class AllFrameSelector:
  """Selects all of the frames of a video."""

  def __call__(self, frame_container):
    """Yields all frames of the specified VideoFrameContainer,
    sorted by ascending frame_id."""
    for frame in sorted(frame_container.frames(), key=lambda x: x[0]):
      print "--> AllFrameSelector selects frame " + str(frame[0]) + ", quality " + str(frame[2][0]) # TODO: remove debug
      yield frame[1]

class FirstNFrameSelector:
  """Selects the first N frames of a video and discards the rest."""
  
  def __init__(self, N):
    self._N = N

  def __call__(self, frame_container):
    """Yields the first N frames of the specified VideoFrameContainer. The video must contain at least N frames, otherwise the behaviour is unspecified."""
    sorted_frames = sorted(frame_container.frames(), key=lambda x: x[0])
    for n in range(self._N):
      print "--> FirstNFrameSelector selects frame " + str(sorted_frames[n][0]) + ", quality " + str(sorted_frames[n][2][0]) # TODO: remove debug
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
      print "--> QualityNFrameSelector selects frame " + str(sorted_frames[n][0]) + ", quality " + str(sorted_frames[n][2][k]) # TODO: remove debug
      yield sorted_frames[n][1]
