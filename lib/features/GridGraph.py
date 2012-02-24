#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob
import numpy

class GridGraph:
  """Extracts grid graphs from the images"""
  
  def __init__(self, setup):
    #   generate extractor machine
    gabor_kwargs = {}
    if hasattr(setup, 'GABOR_DIRECTIONS'): gabor_kwargs['number_of_angles'] = setup.GABOR_DIRECTIONS
    if hasattr(setup, 'GABOR_SCALES'): gabor_kwargs['number_of_scales'] = setup.GABOR_SCALES
    if hasattr(setup, 'GABOR_SIGMA'): gabor_kwargs['sigma'] = setup.GABOR_SIGMA
    if hasattr(setup, 'GABOR_K_MAX'): gabor_kwargs['k_max'] = setup.GABOR_K_MAX
    if hasattr(setup, 'GABOR_K_FAC'): gabor_kwargs['k_fac'] = setup.GABOR_K_FAC
    self.m_gwt = bob.ip.GaborWaveletTransform(**gabor_kwargs)

    if hasattr(setup, 'COUNT_BETWEEN_EYES'):
      # compute eye positions from image preprocessing setup
      left_eye = [setup.CROP_OH, setup.CROP_OW + setup.CROP_EYES_D/2 + setup.CROP_EYES_D%2]
      right_eye = [setup.CROP_OH, setup.CROP_OW - setup.CROP_EYES_D/2]
      
      self.m_graph_machine = bob.machine.GaborGraphMachine(
              left_eye, right_eye, 
              setup.COUNT_BETWEEN_EYES, setup.COUNT_ALONG_EYES, setup.COUNT_ABOVE_EYES, setup.COUNT_BELOW_EYES
      )
    elif hasattr(setup, 'FIRST'):
      # take the specified positions
      self.m_graph_machine = bob.machine.GaborGraphMachine(setup.FIRST, setup.LAST, setup.STEP)
    else:
      raise "The setup of the Grid graph is unknown."
      
    self.m_jet_image = None
    self.m_extract_phases = setup.extract_phases
    self.m_normalize_jets = setup.normalize_jets

    # preallocate memory for the face graph    
    if self.m_extract_phases:
      self.m_face_graph = numpy.ndarray((self.m_graph_machine.number_of_nodes, 2, self.m_gwt.number_of_kernels), 'float64')
    else:
      self.m_face_graph = numpy.ndarray((self.m_graph_machine.number_of_nodes, self.m_gwt.number_of_kernels), 'float64')
    
    
    
  def __call__(self, image):
    if self.m_jet_image == None or self.m_jet_image.shape[0:1] != image.shape:
      # create jet image
      self.m_jet_image = self.m_gwt.jet_image(image, self.m_extract_phases)
      
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

    # return the extracted face graph     
    return self.m_face_graph

