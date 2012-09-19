#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy

from .. import utils

class Eigenface:
  """Extracts grid graphs from the images"""

  def __init__(self, setup):
    #   generate extractor machine
    self.m_config = setup

  def __linearize__(self, image):
    return numpy.reshape(image, image.size)

  def train(self, image_list, extractor_file):
    """Trains the eigenface extractor using the given list of training images"""
    # Initializes an arrayset for the data
    data = bob.io.Arrayset()
    for k in sorted(image_list.keys()):
      # Appends in the arrayset
      data.append(self.__linearize__(image_list[k]))

    utils.info("  -> Training LinearMachine using PCA (SVD)")
    t = bob.trainer.SVDPCATrainer()
    self.m_machine, __eig_vals = t.train(data)
    # Machine: get shape, then resize
    self.m_machine.resize(self.m_machine.shape[0], self.m_config.SUBSPACE_DIMENSION)
    self.m_machine.save(bob.io.HDF5File(extractor_file, "w"))


  def load(self, extractor_file):
    # read PCA projector
    self.m_machine = bob.machine.LinearMachine(bob.io.HDF5File(extractor_file))
    # Allocates an array for the projected data
    self.m_projected_feature = numpy.ndarray(self.m_machine.shape[1], numpy.float64)

  def __call__(self, image):
    """Projects the data using the stored covariance matrix"""
    # Projects the data
    self.m_machine(self.__linearize__(image), self.m_projected_feature)
    # return the projected data
    return self.m_projected_feature

