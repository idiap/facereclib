#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""Features for face recognition"""

import numpy,math
import bob
from DCT import DCTBlocks
from .. import utils

class DCTBlocksVideo(DCTBlocks):

  def __init__(self, config):
    DCTBlocks.__init__(self, config)


  def read_feature(self, filename):
    """Read video.FrameContainer containing features extracted from each frame"""
    return utils.video.FrameContainer(str(filename))


  def __call__(self, frame_container):
    """Returns local DCT features computed from each frame in the input video.FrameContainer"""

    output_frame_container = utils.video.FrameContainer()
    for (frame_id, image, quality) in frame_container.frames():
      frame_dcts = DCTBlocks._dct_features(self,image)
      output_frame_container.add_frame(frame_id,frame_dcts,quality)

    return output_frame_container

