#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy
from .Extractor import Extractor

class GridGraph (Extractor):
  """Extracts grid graphs from the images"""

  def __init__(self, setup):
    Extractor.__init__(self)
    #   generate extractor machine
    self.m_gwt = bob.ip.GaborWaveletTransform(
          number_of_angles = setup.GABOR_DIRECTIONS,
          number_of_scales = setup.GABOR_SCALES,
          sigma = setup.GABOR_SIGMA,
          k_max = setup.GABOR_MAXIMUM_FREQUENCY,
          k_fac = setup.GABOR_FREQUENCY_STEP,
          pow_of_k = setup.GABOR_POWER_OF_K,
          dc_free = setup.GABOR_DC_FREE
    )

    if hasattr(setup, 'COUNT_BETWEEN_EYES'):
      # compute eye positions from image preprocessing setup
      self.m_graph_machine = bob.machine.GaborGraphMachine(
              righteye = setup.RIGHT_EYE_POS,
              lefteye = setup.LEFT_EYE_POS,
              between = setup.NODE_COUNT_BETWEEN_EYES,
              along = setup.NODE_COUNT_ALONG_EYES,
              above = setup.NODE_COUNT_ABOVE_EYES,
              below = setup.NODE_COUNT_BELOW_EYES
      )
    elif hasattr(setup, 'FIRST_NODE'):
      # take the specified positions
      self.m_graph_machine = bob.machine.GaborGraphMachine(setup.FIRST_NODE, setup.LAST_NODE, setup.NODE_DISTANCE)
    else:
      raise "The setup of the Grid graph is unknown."

    self.m_jet_image = None
    self.m_normalize_jets = setup.NORMALIZE_GABOR_JETS
    if isinstance(setup.EXTRACT_GABOR_PHASES, bool):
      self.m_extract_phases = setup.EXTRACT_GABOR_PHASES
      self.m_inline_phases = False
    else:
      self.m_extract_phases = setup.EXTRACT_GABOR_PHASES == 'inline'
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

