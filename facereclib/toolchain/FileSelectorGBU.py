#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import os
from .. import utils
import bob
import xml.sax
import pickle
import numpy

class ImageInformation:
  """This class holds all the information available for one image"""
  
  def __init__(self, identity, filename, name):
    """Initializes the image information"""
    self.m_filename = filename
    self.m_identity = identity
    self.m_name = name
    self.m_eye_position = None

  def set_eye_position(self, eye_pos):
    """Setter fucntion for the eye position"""
    self.m_eye_position = [int(e) for e in eye_pos]
    
  def eye_position(self):
    """Getter function for the eye position"""
    return self.m_eye_position
    
  def file_id(self):
    """Get the file ID from the image name"""
    return os.path.splitext(os.path.basename(self.m_filename))[0]
  
  def file_name(self, extension = None):
    """Return the file name, possibly with the extension replaced by the given one."""
    if extension:
      return self.file_id() + extension
    else:
      return self.m_filename
    
  def identity(self):
    return self.m_identity


class XmlFileReaderGBU (xml.sax.handler.ContentHandler):
  def __init__(self):
    pass
  
  def startDocument(self):
    self.m_image_list = []
    
  def endDocument(self):
    pass
    
  def startElement(self, name, attrs):
    if name == 'biometric-signature':
      self.m_signature = attrs['name']
    elif name == 'presentation':
      self.m_file_name = attrs['file-name']
      self.m_name = attrs['name']
    else:
      pass

  def endElement(self, name):
    if name == 'biometric-signature':
      self.m_image_list.append(ImageInformation(self.m_signature, self.m_file_name, self.m_name))
      self.m_signature = None
      self.m_file_name = None
      self.m_name = None
    else:
      pass

  

class FileSelectorGBU:
  """This class provides shortcuts for selecting different files for different stages of the verification process"""
  
  def __read_list__(self, xml_file, eye_file = None):
    """Reads the xml list and attaches the eye files, if given"""
    # create xml reading instance
    handler = XmlFileReaderGBU()
    xml.sax.parse(xml_file, handler)
    image_list = handler.m_image_list

    if eye_file:
      # generate temporary dictionary for faster read of the eye position file    
      image_dict={}
      for image in image_list:
        image_dict[image.file_id()] = image
      
      # read the eye position list
      f = open(eye_file)
      for line in f:
        entries=line.split(',')
        assert len(entries) == 5
        name = os.path.splitext(os.path.basename(entries[0]))[0]
        # test if these eye positions belong to any file of this list
        if name in image_dict:
          image_dict[name].set_eye_position(entries[1:])

    return image_list
  
  def __generate_full_mask__(self):
    """Generates a complete mask that is used to specify which pairs to be used for comparison"""
    mask = numpy.ones((len(self.m_query), len(self.m_target)), dtype = numpy.uint8) * 127
    for query_index in range(len(self.m_query)):
      for target_index in range(len(self.m_target)):
        if self.m_query[query_index].identity() == self.m_target[target_index].identity():
          mask[query_index,target_index] = 255
          
    return mask
  
  def __init__(self, config):
    """Initialize the file selector object with the current configuration"""
    self.m_config = config

    # read the three sets of images
    print "Reading XML lists"
    self.m_training = self.__read_list__(config.training, config.eye_file)
    self.m_target = self.__read_list__(config.target, config.eye_file)
    self.m_query = self.__read_list__(config.query, config.eye_file)
    if hasattr(config, 'mask'):
      import pyvision
      self.m_mask = pyvision.BEEDistanceMatrix(config.mask).matrix
    else:
      self.m_mask = self.__generate_full_mask__()
    
    
  def __generate_list__(self, set, directory, extension = None, use_protocol_subdir = True):
    """Generates the list of file names for the given set"""
    list = []
    if set == 'training':
      image_list = self.m_training
      sub_dir = '.' 
    else: 
      if set == 'target':
        image_list = self.m_target
      else: 
        image_list = self.m_query
      sub_dir = self.m_config.protocol if use_protocol_subdir else '.'
    for i in image_list:
      list.append(os.path.join(directory, sub_dir, i.file_name(extension)))
    return list
  
  def __ids__(self, type):
    ids = set()
    image_list = self.m_training if type == 'training' else self.m_target if type == 'target' else self.m_query
    for i in image_list:
      id.add(i.file_id())
    
    
  ### Original images and preprocessing
  def image_list(self, set):
    """Returns the list of original images that should be used for image preprocessing"""
    return self.__generate_list__(set, self.m_config.img_input_dir, use_protocol_subdir = False)
    
  def eye_position_list(self, set):
    """Returns the list of eye positions"""
    list = []
    image_list = self.m_training if set == 'training' else self.m_target if set == 'target' else self.m_query
    for i in image_list:
      list.append(i.m_eye_position)
    return list
   
  def preprocessed_image_list(self, set):
    """Returns the list of preprocessed images and assures that the normalized image path is existing"""
    dir = os.path.join(self.m_config.preprocessed_dir, set)
    utils.ensure_dir(dir)
    return self.__generate_list__(set, dir, self.m_config.default_extension)

  def feature_list(self, set):
    """Returns the list of features and assures that the feature path is existing"""
    dir = os.path.join(self.m_config.features_dir, set)
    utils.ensure_dir(dir)
    return self.__generate_list__(set, dir, self.m_config.default_extension)

  def projected_list(self, set):
    """Returns the list of projected features and assures that the projected feature path is existing"""
    dir = os.path.join(self.m_config.projected_dir, set)
    utils.ensure_dir(dir)
    return self.__generate_list__(set, dir, self.m_config.default_extension)
    
  def training_feature_list(self, dir_type):
    """Returns the list of features and assures that the feature path is existing"""
    # get the type of directory that is required
    if dir_type == 'preprocessed': 
      cur_dir = self.m_config.preprocessed_dir
    elif dir_type == 'features': 
      cur_dir = self.m_config.features_dir 
    elif dir_type == 'projected': 
      cur_dir = self.m_config.projected_dir
    # iterate over all training model ids
    training_filenames = {}
    for i in self.m_training:
      name = os.path.join(cur_dir, 'training', i.file_name(self.m_config.default_extension))
      # to be compatible with the tools, we have to mimic the dictionary behaviour
      training_filenames[i.file_id()] = name
    # return the dictionary of training file names
    return training_filenames


  def training_feature_list_by_models(self, dir_type):
    """Returns the list of training features, which is split up by the client ids."""
    # get the type of directory that is required
    if dir_type == 'preprocessed': 
      cur_dir = self.m_config.preprocessed_dir
    elif dir_type == 'features': 
      cur_dir = self.m_config.features_dir 
    elif dir_type == 'projected': 
      cur_dir = self.m_config.projected_dir
    # iterate over all training model ids
    training_filenames = {}
    for i in self.m_training:
      name = os.path.join(cur_dir, 'training', i.file_name(self.m_config.default_extension))
      # to be compatible with the tools, we have to mimic the dictionary behaviour
      if i.identity() in training_filenames:
        training_filenames[i.identity()][i.file_id()] = name
      else:
        training_filenames[i.identity()] = {i.file_id() : name}
    # return the dictionary of training file names
    return training_filenames
    

  def extractor_file(self):
    """Returns the file where to save the trainined extractor model to"""
    utils.ensure_dir(os.path.dirname(self.m_config.extractor_file))
    return self.m_config.extractor_file

  def projector_file(self):
    """Returns the file where to save the trained model"""
    utils.ensure_dir(os.path.dirname(self.m_config.projector_file))
    return self.m_config.projector_file
    
  def enroler_file(self):
    """Returns the name of the file that includes the model trained for enrolment"""
    utils.ensure_dir(os.path.dirname(self.m_config.enroler_file))
    return self.m_config.enroler_file
    
    
  def model_indices(self):
    """Returns the list of model indices from the given group"""
    return range(len(self.m_target))

  def enrol_files(self, index, use_projected_dir):
    """Returns the list of model features (in this case, only one feature per model) used for enrolment of the given model_id from the given group"""
    used_dir = os.path.join(self.m_config.projected_dir if use_projected_dir else self.m_config.features_dir, 'target', self.m_config.protocol)
    return [ os.path.join(used_dir, self.m_target[index].file_name(self.m_config.default_extension)) ]
    
  def model_file(self, model_index):
    """Returns the model file for the given model index"""
    file = os.path.join(self.m_config.model_dir, self.m_target[model_index].file_name(self.m_config.default_extension))
    utils.ensure_dir(os.path.dirname(file))
    return file
  
    
  def probe_files(self, use_projected_dir):
    """Returns the probe files used to compute the raw scores"""
    dir = os.path.join(self.m_config.projected_dir if use_projected_dir else self.m_config.features_dir, 'query')
    utils.ensure_dir(dir)
    return self.__generate_list__('query', dir, self.m_config.default_extension)
    
  
  def queries_for_target(self, target_index):
    """Returns a list of query files for the given target index"""
    queries=[]
    for query_index in range(len(self.m_query)):
      if self.m_mask[query_index, target_index]:
        # assert that mask value 255 (i.e. -1) is used when both identities are the same
        if ((self.m_mask[query_index,target_index] == 127) == (self.m_query[query_index].identity() == self.m_target[target_index].identity())):
          print "Warning: mask value", self.m_mask[target_index,query_index], "is wrong for pair", self.m_query[query_index].identity(), self.m_target[target_index].identity()

        queries.append((
                query_index,
                self.m_query[query_index].identity(),
                self.m_target[target_index].identity()
                ))
    return queries
        
  def model_score_file(self, index):
    """Returns the score file where to write the target scores into"""
    dir = os.path.join(self.m_config.score_dir, 'models')
    utils.ensure_dir(dir)
    return os.path.join(dir, self.m_target[index].file_id() + '.txt')
    
  def result_file(self):
    norm_dir = self.m_config.score_dir
    utils.ensure_dir(norm_dir)
    return os.path.join(norm_dir, "scores")
   
    
