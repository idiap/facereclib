#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.ip.gabor
import bob.io.base

import numpy
import math
from .Extractor import Extractor

class GridGraph (Extractor):
  """Extracts grid graphs from the images"""

  def __init__(
      self,
      # Gabor parameters
      gabor_directions = 8,
      gabor_scales = 5,
      gabor_sigma = 2. * math.pi,
      gabor_maximum_frequency = math.pi / 2.,
      gabor_frequency_step = math.sqrt(.5),
      gabor_power_of_k = 0,
      gabor_dc_free = True,

      # what kind of information to extract
      normalize_gabor_jets = True,

      # setup of the aligned grid
      eyes = None, # if set, the grid setup will be aligned to the eye positions {'leye' : LEFT_EYE_POS, 'reye' : RIGHT_EYE_POS},
      nodes_between_eyes = 4,
      nodes_along_eyes = 2,
      nodes_above_eyes = 3,
      nodes_below_eyes = 7,

      # setup of static grid
      node_distance = None,    # one or two integral values
      image_resolution = None, # always two integral values
      first_node = None,       # one or two integral values, or None -> automatically determined
  ):

    # call base class constructor
    Extractor.__init__(
        self,

        gabor_directions = gabor_directions,
        gabor_scales = gabor_scales,
        gabor_sigma = gabor_sigma,
        gabor_maximum_frequency = gabor_maximum_frequency,
        gabor_frequency_step = gabor_frequency_step,
        gabor_power_of_k = gabor_power_of_k,
        gabor_dc_free = gabor_dc_free,
        normalize_gabor_jets = normalize_gabor_jets,
        eyes = eyes,
        nodes_between_eyes = nodes_between_eyes,
        nodes_along_eyes = nodes_along_eyes,
        nodes_above_eyes = nodes_above_eyes,
        nodes_below_eyes = nodes_below_eyes,
        node_distance = node_distance,
        image_resolution = image_resolution,
        first_node = first_node
    )

    # create Gabor wavelet transform class
    self.m_gwt = bob.ip.gabor.Transform(
        number_of_scales = gabor_scales,
        number_of_directions = gabor_directions,
        sigma = gabor_sigma,
        k_max = gabor_maximum_frequency,
        k_fac = gabor_frequency_step,
        power_of_k = gabor_power_of_k,
        dc_free = gabor_dc_free
    )

    # create graph extractor
    if eyes is not None:
      self.m_graph = bob.ip.gabor.Graph(
          righteye = [int(e) for e in eyes['reye']],
          lefteye = [int(e) for e in eyes['leye']],
          between = int(nodes_between_eyes),
          along = int(nodes_along_eyes),
          above = int(nodes_above_eyes),
          below = int(nodes_below_eyes)
      )
    else:
      if node_distance is None or image_resolution is None:
        raise ValueError("Please specify either 'eyes' or the grid parameters 'first_node', 'last_node', and 'node_distance'!")
      if isinstance(node_distance, (int, float)):
         node_distance = (int(node_distance), int(node_distance))
      if first_node is None:
        first_node = [0,0]
        for i in (0,1):
          offset = int((image_resolution[i] - int(image_resolution[i]/node_distance[i])*node_distance[i]) / 2)
          if offset < node_distance[i]//2: # This is not tested, but should ALWAYS be the case.
            offset += node_distance[i]//2
          first_node[i] = offset
      last_node = tuple([int(image_resolution[i] - max(first_node[i],1)) for i in (0,1)])

      # take the specified nodes
      self.m_graph = bob.ip.gabor.Graph(
          first = first_node,
          last = last_node,
          step = node_distance
      )

    self.m_normalize_jets = normalize_gabor_jets
    self.m_trafo_image = None

  def __call__(self, image):
    if self.m_trafo_image is None or self.m_trafo_image.shape[1:3] != image.shape:
      # create trafo image
      self.m_trafo_image = numpy.ndarray((self.m_gwt.number_of_wavelets, image.shape[0], image.shape[1]), numpy.complex128)

    # perform Gabor wavelet transform
    self.m_gwt.transform(image, self.m_trafo_image)

    # extract face graph
    jets = self.m_graph.extract(self.m_trafo_image)

    # normalize the Gabor jets of the graph only
    if self.m_normalize_jets:
      [j.normalize() for j in jets]

    # return the extracted face graph
    return jets

  def save_feature(self, feature, feature_file):
    feature_file = feature_file if isinstance(feature_file, bob.io.base.HDF5File) else bob.io.base.HDF5File(feature_file, 'w')
    bob.ip.gabor.save_jets(feature, feature_file)

  def read_feature(self, feature_file):
    return bob.ip.gabor.load_jets(bob.io.base.HDF5File(feature_file))
