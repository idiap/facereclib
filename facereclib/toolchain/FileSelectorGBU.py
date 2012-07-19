#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import os
from .. import utils
import bob

class FileSelectorGBU:
  """This class provides shortcuts for selecting different files for different stages of the verification process"""
  
  def __init__(self, config, db):
    """Initialize the file selector object with the current configuration"""
    self.m_config = config
    self.m_db_options = db 
    self.m_db = db.db
    
  def __options__(self, set, name = None):
    opts={'type':'gbu'}
    # get the options from the db settings
    if name != None and hasattr(self.m_db_options, name):
      opts.update(eval('self.m_db_options.'+name))
    if set == 'training':
      opts['groups'] = 'world'
    else:
      opts['groups'] = 'dev'
      if set == 'target':
        opts['purposes']='enrol'
      else:
        opts['purposes']='probe'
    return opts
    
  ### Original images and preprocessing
  def original_image_list(self, set):
    """Returns the list of original images that should be used for image preprocessing"""
    opts=self.__options__(set, 'all_files_options')
    return self.m_db.files(directory=self.m_config.img_input_dir, extension=self.m_config.img_input_ext, protocol=self.m_config.protocol, **opts)
    
  def eye_position_list(self, set):
    """Returns the list of eye positions"""
    opts=self.__options__(set, 'all_files_options')
    # query the DB
    objects = self.m_db.objects(**opts)
    eyes={}
    for k,v in objects.iteritems():
      eyes[k] = v[2]
    return eyes
   
  def preprocessed_image_list(self, set):
    """Returns the list of preprocessed images and assures that the normalized image path is existing"""
    opts=self.__options__(set, 'all_files_options')
    return self.m_db.files(directory=self.m_config.preprocessed_dir, extension=self.m_config.default_extension, protocol=self.m_config.protocol, **opts)

  def feature_list(self, set):
    """Returns the list of features and assures that the feature path is existing"""
    opts=self.__options__(set, 'all_files_options')
    return self.m_db.files(directory=self.m_config.features_dir, extension=self.m_config.default_extension, protocol=self.m_config.protocol, **opts)

  def projected_list(self, set):
    """Returns the list of projected features and assures that the projected feature path is existing"""
    opts=self.__options__(set, 'all_files_options')
    return self.m_db.files(directory=self.m_config.projected_dir, extension=self.m_config.default_extension, protocol=self.m_config.protocol, **opts)
    
  def training_feature_list(self, dir_type):
    """Returns the list of features and assures that the feature path is existing"""
    # get the type of directory that is required
    if dir_type == 'preprocessed': 
      cur_dir = self.m_config.preprocessed_dir
    elif dir_type == 'features': 
      cur_dir = self.m_config.features_dir 
    elif dir_type == 'projected': 
      cur_dir = self.m_config.projected_dir
    # query the database
    return self.m_db.files(directory=cur_dir, extension=self.m_config.default_extension, protocol=self.m_config.protocol, **self.__options__('training', 'world_extractor_options'))  


  def training_feature_list_by_clients(self, dir_type):
    """Returns the list of training features, which is split up by the client ids."""
    # get the type of directory that is required
    if dir_type == 'preprocessed': 
      cur_dir = self.m_config.preprocessed_dir 
      cur_world_options = self.__options__('training', 'world_extractor_options')
    elif dir_type == 'features': 
      cur_dir = self.m_config.features_dir 
      cur_world_options = self.__options__('training', 'world_projector_options')
    elif dir_type == 'projected': 
      cur_dir = self.m_config.projected_dir
      cur_world_options = self.__options__('training', 'world_enroler_options')
    # in this case, we need the type 'multi' (the default) 
    # since we want to get several files per client
    del cur_world_options['type']
    # iterate over all training model ids
    training_clients = self.m_db.clients(**cur_world_options)
    training_files = {}
    for client in training_clients:
      # collect training features for current model id
      client_files = self.m_db.files(directory=cur_dir, extension=self.m_config.default_extension, protocol=self.m_config.protocol, model_ids=(client,), **cur_world_options) 
      # add this model to the list
      training_files[client] = client_files
    # return the list of models
    return training_files
    

  def extractor_file(self):
    """Returns the file where to save the trained extractor model to"""
    utils.ensure_dir(os.path.dirname(self.m_config.extractor_file))
    return self.m_config.extractor_file

  def projector_file(self):
    """Returns the file where to save the trained model"""
    utils.ensure_dir(os.path.dirname(self.m_config.projector_file))
    return self.m_config.projector_file
    
  def enroler_file(self):
    """Returns the name of the file that includes the model trained for enrollment"""
    utils.ensure_dir(os.path.dirname(self.m_config.enroler_file))
    return self.m_config.enroler_file
    
    
  def model_ids(self):
    """Returns the list of model indices (one model per file)"""
    return self.m_db.models(type='gbu', groups='dev', protocol=self.m_config.protocol)

  def enrol_files(self, model_id, use_projected_dir):
    """Returns the list of model features (in this case, only one feature per model) used for enrollment of the given model_id from the given group"""
    used_dir = self.m_config.projected_dir if use_projected_dir else self.m_config.features_dir
    return self.m_db.files(directory=used_dir, extension=self.m_config.default_extension, groups='dev', purposes='enrol', protocol = self.m_config.protocol, model_ids=(model_id,), type='gbu')
    
  def model_file(self, model_id):
    """Returns the model file for the given model index"""
    file = os.path.join(self.m_config.model_dir,  str(model_id) + self.m_config.default_extension)
    utils.ensure_dir(os.path.dirname(file))
    return file
  
  def probe_files(self, use_projected_dir):
    """Returns the probe files used to compute the raw scores"""
    used_dir = self.m_config.projected_dir if use_projected_dir else self.m_config.features_dir
    utils.ensure_dir(used_dir)
    return self.m_db.files(directory=used_dir, extension=self.m_config.default_extension, groups='dev', purposes='probe', protocol = self.m_config.protocol)
  
  def client_id_for_id(self, id):
    """Returns the client id that is connected to the given id, which might be a model or a probe id"""
    return self.m_db.get_client_id_from_file_id(id)
  
  def model_score_file(self, model_id):
    """Returns the score file where to write the target scores into"""
    dir = os.path.join(self.m_config.score_dir, 'models')
    utils.ensure_dir(dir)
    return os.path.join(dir, model_id + '.txt')
    
  def result_file(self):
    norm_dir = self.m_config.score_dir
    utils.ensure_dir(norm_dir)
    return os.path.join(norm_dir, "scores")
   
    
