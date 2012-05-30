#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch> 
# @date: Thu May 24 10:41:42 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import bob
import numpy
from .. import utils

class TanTriggs:
  """Crops the face and applies Tan-Triggs algorithm"""

  def __init__(self, config):
    self.m_config = config
    # prepare image normalization
    self.m_fen = bob.ip.FaceEyesNorm(config.CROP_EYES_D, config.CROP_H, config.CROP_W, config.CROP_OH, config.CROP_OW)
    self.m_fen_image = numpy.ndarray((config.CROP_H, config.CROP_W), numpy.float64) 
    self.m_tan = bob.ip.TanTriggs(config.GAMMA, config.SIGMA0, config.SIGMA1, config.SIZE, config.THRESHOLD, config.ALPHA)
    self.m_tan_image = numpy.ndarray((config.CROP_H, config.CROP_W), numpy.float64)

  def __call__(self, input_file, output_file, eye_pos = None):
    """Reads the input image, normalizes it according to the eye positions, and writes the resulting image"""
    image = bob.io.load(str(input_file))
    # convert to grayscale
    if(image.ndim == 3):
      image = bob.ip.rgb_to_gray(image)
      
    # perform image normalization
    if eye_pos == None:
      self.m_fen_image = image
      if self.m_tan_image.shape != image.shape:
        self.m_tan_image =  numpy.ndarray(image.shape, numpy.float64)
    else: 
      self.m_fen(image, self.m_fen_image, eye_pos[1], eye_pos[0], eye_pos[3], eye_pos[2])
      
    # perform Tan-Triggs
    self.m_tan(self.m_fen_image, self.m_tan_image)
    
    # save output image    
    bob.io.save(self.m_tan_image, output_file)
    
    
class StaticTanTriggs:
  """Crops the face to FIXED eye positions and applies Tan-Triggs algorithm"""

  def __init__(self, config):
    self.m_config = config
    # prepare image normalization
    self.m_fen = bob.ip.FaceEyesNorm(config.CROP_EYES_D, config.CROP_H, config.CROP_W, config.CROP_OH, config.CROP_OW)
    self.m_fen_image = numpy.ndarray((config.CROP_H, config.CROP_W), numpy.float64) 
    self.m_tan = bob.ip.TanTriggs(config.GAMMA, config.SIGMA0, config.SIGMA1, config.SIZE, config.THRESHOLD, config.ALPHA)
    self.m_tan_image = numpy.ndarray((config.CROP_H, config.CROP_W), numpy.float64)

  def __call__(self, input_file, output_file, eye_pos = None):
    """Reads the input image, normalizes it according to the eye positions, and writes the resulting image"""
    image = bob.io.load(str(input_file))
    # convert to grayscale
    if(image.ndim == 3):
      image = bob.ip.rgb_to_gray(image)
      
    # perform image normalization
    self.m_fen(image, self.m_fen_image, self.m_config.RIGHT_EYE[0], self.m_config.RIGHT_EYE[1], self.m_config.LEFT_EYE[0], self.m_config.LEFT_EYE[1])
      
    # perform Tan-Triggs
    self.m_tan(self.m_fen_image, self.m_tan_image)
    
    # save output image    
    bob.io.save(self.m_tan_image, output_file)
    
    
class TanTriggsVideo:
  """Applies the Tan-Triggs algorithm to each frame in a video"""

  def __init__(self, config):
    self.m_config = config
    # prepare image normalization
    self.m_tan = bob.ip.TanTriggs(config.GAMMA, config.SIGMA0, config.SIGMA1, config.SIZE, config.THRESHOLD, config.ALPHA)

  #def __call__(self, input_file, output_file, eye_pos = None): # Note: eye_pos not supported. Videos are assumed to be already 
  def __call__(self, input_file, output_file, eye_pos = None):
    """For each frame in the VideoFrameContainer (read from input_file) applies the Tan-Triggs algorithm, then writes the resulting VideoFrameContainer to output_file. NOTE: eye_pos is ignored even if specified."""
    # Read input
    frame_container = utils.VideoFrameContainer(str(input_file))

    # Process each frame
    output_frame_container = utils.VideoFrameContainer()
    for (frame_id, image) in frame_container.frames():
      
      # Convert to grayscale if it seems necessary
      if(image.ndim == 3):
        image = bob.ip.rgb_to_gray(image)
        
      # Perform Tan-Triggs and store result
      self.m_tan_image = numpy.ndarray(image.shape, numpy.float64)
      self.m_tan(image, self.m_tan_image)
      output_frame_container.add_frame(frame_id,self.m_tan_image)
      
    # Save output image    
    output_frame_container.save(output_file)

