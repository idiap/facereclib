#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

"""A frame selector filters a subset of the frames from an input VideoFrameContainer"""
# TODO later:
# from Quality import QualityFrameSelector
# from Fancy import FancyFrameSelector
# ...

import numpy
import bob

class AllFrameSelector:
  """Selects all of the frames of a video"""

  def __call__(self, frame_container):
    """Yields all frames of the specified VideoFrameContainer"""
    for data in frame_container.frames_data():
      yield data


class FirstNFrameSelector:
  """Selects the first N frames of a video and discards the rest"""
  
  def __init__(self, N):
    self._N = N

  def __call__(self, frame_container):
    """Yields the first N frames of the specified VideoFrameContainer"""
    for n in range(self._N):
      yield frame_container.frame_data(n)

