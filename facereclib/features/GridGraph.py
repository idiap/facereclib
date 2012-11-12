#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
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
      extract_gabor_phases = True,

      # setup of the aligned grid
      eyes = None, # if set, the grid setup will be aligned to the eye positions {'leye' : LEFT_EYE_POS, 'reye' : RIGHT_EYE_POS},
      nodes_between_eyes = 4,
      nodes_along_eyes = 2,
      nodes_above_eyes = 3,
      nodes_below_eyes = 7,

      # setup of static grid
      first_node = None,    # one or two integral values
      node_distance = None, # always two integral values
      last_node = None      # one or two integral values
  ):

    # call base class constructor
    Extractor.__init__(self)

    # create Gabor wavelet transform class
    self.m_gwt = bob.ip.GaborWaveletTransform(
        number_of_scales = gabor_scales,
        number_of_angles = gabor_directions,
        sigma = gabor_sigma,
        k_max = gabor_maximum_frequency,
        k_fac = gabor_frequency_step,
        pow_of_k = gabor_power_of_k,
        dc_free = gabor_dc_free
    )

    # create graph extractor
    if eyes is not None:
      self.m_graph_machine = bob.machine.GaborGraphMachine(
          righteye = eyes['reye'],
          lefteye = eyes['leye'],
          between = nodes_between_eyes,
          along = nodes_along_eyes,
          above = nodes_above_eyes,
          below = nodes_below_eyes
      )
    else:
      if first_node is None or node_distance is None or last_node is None:
        raise ValueError("Please specify either 'eyes' or the grid parameters 'first_node', 'last_node', and 'node_distance'!")
      # take the specified nodes
      self.m_graph_machine = bob.machine.GaborGraphMachine(
          first = (first_node, first_node) if isinstance(first_node, int) else first_node,
          last = last_node,
          step = (node_distance, node_distance) if isinstance(node_distance, int) else node_distance
      )

    self.m_jet_image = None
    self.m_normalize_jets = normalize_gabor_jets
    if isinstance(extract_gabor_phases, bool):
      self.m_extract_phases = extract_gabor_phases
      self.m_inline_phases = False
    else:
      self.m_extract_phases = extract_gabor_phases == 'inline'
      self.m_inline_phases = self.m_extract_phases

    # preallocate memory for the face graph
    if self.m_extract_phases:
      self.m_face_graph = numpy.ndarray((self.m_graph_machine.number_of_nodes, 2, self.m_gwt.number_of_kernels), 'float64')
      if self.m_inline_phases:
        self.m_reshaped_graph = numpy.ndarray((self.m_graph_machine.number_of_nodes, 2 * self.m_gwt.number_of_kernels), 'float64')
    else:
      self.m_face_graph = numpy.ndarray((self.m_graph_machine.number_of_nodes, self.m_gwt.number_of_kernels), 'float64')


  def __call__(self, image):
    if self.m_jet_image == None or self.m_jet_image.shape[0:1] != image.shape:
      # create jet image
      self.m_jet_image = self.m_gwt.empty_jet_image(image, self.m_extract_phases)

    # compute jets (Do not normalize the Gabor jets of the whole image)
    self.m_gwt.compute_jets(image, self.m_jet_image, False)
    # extract face graph
    self.m_graph_machine(self.m_jet_image, self.m_face_graph)

    # normalize the Gabor jets of the graph only
    if self.m_normalize_jets:
      for n in range(self.m_face_graph.shape[0]):
        if self.m_extract_phases:
          bob.ip.normalize_gabor_jet(self.m_face_graph[n,:,:])
        else:
          bob.ip.normalize_gabor_jet(self.m_face_graph[n,:])

    if self.m_inline_phases:
      # reshape Gabor jets of the graph, if desired
      for i in range(self.m_graph_machine.number_of_nodes):
        self.m_reshaped_graph[i,:] = self.m_face_graph[i].flatten()
      return self.m_reshaped_graph

    else:
      # return the extracted face graph
      return self.m_face_graph

