#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import numpy

import bob.learn.linear
import bob.io.base

from .Extractor import Extractor
from .. import utils

class Eigenface (Extractor):
  """Extracts grid graphs from the images"""

  def __init__(self, subspace_dimension):
    # We have to register that this function will need a training step
    Extractor.__init__(self, requires_training = True, subspace_dimension = subspace_dimension)
    self.m_subspace_dimension = subspace_dimension

  def train(self, image_list, extractor_file):
    """Trains the eigenface extractor using the given list of training images"""
    # Initializes an array for the data
    data = numpy.vstack([image.flatten() for image in image_list])

    utils.info("  -> Training LinearMachine using PCA (SVD)")
    t = bob.learn.linear.PCATrainer()
    self.m_machine, __eig_vals = t.train(data)
    # Machine: get shape, then resize
    self.m_machine.resize(self.m_machine.shape[0], self.m_subspace_dimension)
    self.m_machine.save(bob.io.base.HDF5File(extractor_file, "w"))


  def load(self, extractor_file):
    # read PCA projector
    self.m_machine = bob.learn.linear.Machine(bob.io.base.HDF5File(extractor_file))
    # Allocates an array for the projected data
    self.m_projected_feature = numpy.ndarray(self.m_machine.shape[1], numpy.float64)

  def __call__(self, image):
    """Projects the data using the stored covariance matrix"""
    # Projects the data
    self.m_machine(image.flatten(), self.m_projected_feature)
    # return the projected data
    return self.m_projected_feature

