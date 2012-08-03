#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import os
from .. import utils
import bob

class FileSelectorZT:
  """This class provides shortcuts for selecting different files for different stages of the verification process"""
  
  def __init__(self, config, db):
    """Initialize the file selector object with the current configuration"""
    self.m_config = config
    self.m_db_options = db 
    self.m_db = db.db
    
  def __options__(self, name):
    """Returnes the options specified by the database, if available"""
    if hasattr(self.m_db_options, name):
      return eval('self.m_db_options.'+name)
    else:
      return {}
    
  ### Original images and preprocessing
  def original_image_list(self):
    """Returns the list of original images that can be used for image preprocessing"""
    return self.m_db.files(directory=self.m_config.img_input_dir, extension=self.m_config.img_input_ext, protocol=self.m_config.protocol, **self.__options__('all_files_options'))
    
  def annotation_list(self):
    """Returns the list of annotation files, if any (else None)"""
    if not hasattr(self.m_config, 'pos_input_dir') or self.m_config.pos_input_dir == None:
      return None

    return self.m_db.files(directory=self.m_config.pos_input_dir, extension=self.m_config.pos_input_ext, protocol=self.m_config.protocol, **self.__options__('all_files_options'))
    
  def preprocessed_image_list(self):
    """Returns the list of preprocessed images and assures that the normalized image path is existing"""
    utils.ensure_dir(self.m_config.preprocessed_dir)
    return self.m_db.files(directory=self.m_config.preprocessed_dir, extension=self.m_config.default_extension, protocol=self.m_config.protocol, **self.__options__('all_files_options'))

  def feature_list(self):
    """Returns the list of features and assures that the feature path is existing"""
    utils.ensure_dir(self.m_config.features_dir)
    return self.m_db.files(directory=self.m_config.features_dir, extension=self.m_config.default_extension, protocol=self.m_config.protocol, **self.__options__('all_files_options'))

  ### Training and projecting features
  def training_image_list(self):
    """Returns the list of images that should be used for extractor training"""
    return self.m_db.files(directory=self.m_config.preprocessed_dir, extension=self.m_config.default_extension, protocol=self.m_config.protocol, groups='world', **self.__options__('world_extractor_options'))  

  def training_feature_list(self):
    """Returns the list of features that should be used for projector training"""
    return self.m_db.files(directory=self.m_config.features_dir, extension=self.m_config.default_extension, protocol=self.m_config.protocol, groups='world', **self.__options__('world_projector_options'))  

  def training_feature_list_by_clients(self, dir_type, step):
    """Returns the list of training features, which is split up by the client ids."""
    # get the type of directory that is required
    if dir_type == 'preprocessed': 
      cur_dir = self.m_config.preprocessed_dir 
    elif dir_type == 'features': 
      cur_dir = self.m_config.features_dir 
    elif dir_type == 'projected': 
      cur_dir = self.m_config.projected_dir

    # if requested, define the subset of training data to be used for this step
    if step == 'train_extractor':
      print "==> getting training_feature_list_by_clients for step: train_extractor" # TODO: remove debug
      cur_world_options = self.__options__('world_extractor_options')
    elif step == 'train_projector':
      print "==> getting training_feature_list_by_clients for step: train_projector" # TODO: remove debug
      cur_world_options = self.__options__('world_projector_options')
    elif step == 'train_enroler':
      print "==> getting training_feature_list_by_clients for step: train_enroler" # TODO: remove debug
      cur_world_options = self.__options__('world_enroler_options')

    # iterate over all training clients
    features_by_clients_options = {}
    if 'subworld' in cur_world_options: features_by_clients_options['subworld'] = cur_world_options['subworld']
    features_by_clients_options.update(self.__options__('features_by_clients_options'))
    train_clients = self.m_db.clients(groups='world', protocol=self.m_config.protocol, **features_by_clients_options)
    training_filenames = {}
    for m in train_clients:
      # collect training features for current client id
      train_data_m = self.m_db.files(directory=cur_dir, extension=self.m_config.default_extension, protocol=self.m_config.protocol, groups='world', model_ids=(m,), **cur_world_options) 
      # add this model to the list
      training_filenames[m] = train_data_m
    # return the list of features which is grouped by client id
    return training_filenames
    

  def extractor_file(self):
    """Returns the file where to save the trainined extractor model to"""
    utils.ensure_dir(os.path.dirname(self.m_config.extractor_file))
    return self.m_config.extractor_file

  def projector_file(self):
    """Returns the file where to save the trained model"""
    utils.ensure_dir(os.path.dirname(self.m_config.projector_file))
    return self.m_config.projector_file
    
  def projected_list(self):
    """Returns the list of projected features and assures that the projected feature path is existing"""
    utils.ensure_dir(self.m_config.projected_dir)
    return self.m_db.files(directory=self.m_config.projected_dir, extension=self.m_config.default_extension, protocol=self.m_config.protocol, **self.__options__('all_files_options'))
    
  ### Enrolment
  def enroler_file(self):
    """Returns the name of the file that includes the model trained for enrolment"""
    utils.ensure_dir(os.path.dirname(self.m_config.enroler_file))
    return self.m_config.enroler_file
    
    
  def model_ids(self, group):
    """Returns the sorted list of model ids from the given group"""
    return sorted(self.m_db.models(protocol=self.m_config.protocol, groups=group))
    
  def enrol_files(self, model_id, group, use_projected_dir):
    """Returns the list of model features used for enrolment of the given model_id from the given group"""
    used_dir = self.m_config.projected_dir if use_projected_dir else self.m_config.features_dir 
    return self.m_db.files(directory=used_dir, extension=self.m_config.default_extension, groups=group, protocol=self.m_config.protocol, model_ids=(model_id,), purposes='enrol')
    
  def model_file(self, model_id, group):
    """Returns the file of the model and assures that the directory exists"""
    model_file = os.path.join(self.m_config.models_dir, group, str(model_id) + self.m_config.default_extension) 
    utils.ensure_dir(os.path.dirname(model_file))
    return model_file
    

  def tmodel_ids(self, group):
    """Returns the sorted list of T-Norm-model ids from the given group"""
    return sorted(self.m_db.tmodels(protocol=self.m_config.protocol, groups=group))
    
  def tenrol_files(self, model_id, group, use_projected_dir):
    """Returns the list of T-model features used for enrolment of the given model_id from the given group"""
    used_dir = self.m_config.projected_dir if use_projected_dir else self.m_config.features_dir 
    return self.m_db.tfiles(directory=used_dir, extension=self.m_config.default_extension, groups=group, protocol=self.m_config.protocol, model_ids=(model_id,))
  
  def tmodel_file(self, model_id, group):
    """Returns the file of the T-Norm-model and assures that the directory exists"""
    tmodel_file = os.path.join(self.m_config.tnorm_models_dir, group, str(model_id) + self.m_config.default_extension) 
    utils.ensure_dir(os.path.dirname(tmodel_file))
    return tmodel_file
    

  ### Probe files and ZT-Normalization  
  def a_file(self, model_id, group):
    a_dir = os.path.join(self.m_config.zt_norm_A_dir, group)
    utils.ensure_dir(a_dir)
    return os.path.join(a_dir, str(model_id) + self.m_config.default_extension)

  def b_file(self, model_id, group):
    b_dir = os.path.join(self.m_config.zt_norm_B_dir, group)
    utils.ensure_dir(b_dir)
    return os.path.join(b_dir, str(model_id) + self.m_config.default_extension)

  def c_file(self, model_id, group):
    c_dir = os.path.join(self.m_config.zt_norm_C_dir, group)
    utils.ensure_dir(c_dir)
    return os.path.join(c_dir, "TM" + str(model_id) + self.m_config.default_extension)

  def c_file_for_model(self, model_id, group):
    c_dir = os.path.join(self.m_config.zt_norm_C_dir, group)
    return os.path.join(c_dir, str(model_id) + self.m_config.default_extension)
    
  def d_file(self, model_id, group):
    d_dir = os.path.join(self.m_config.zt_norm_D_dir, group)
    utils.ensure_dir(d_dir)
    return os.path.join(d_dir, str(model_id) + self.m_config.default_extension)
    
  def d_matrix_file(self, group):
    d_dir = os.path.join(self.m_config.zt_norm_D_dir, group)
    return os.path.join(d_dir, "D" + self.m_config.default_extension)
    
  def d_same_value_file(self, model_id, group):
    d_dir = os.path.join(self.m_config.zt_norm_D_sameValue_dir, group)
    utils.ensure_dir(d_dir)
    return os.path.join(d_dir, str(model_id) + self.m_config.default_extension)

  def d_same_value_matrix_file(self, group):
    d_dir = os.path.join(self.m_config.zt_norm_D_sameValue_dir, group)
    return os.path.join(d_dir, "D_sameValue" + self.m_config.default_extension)
  
  def no_norm_file(self, model_id, group):
    norm_dir = os.path.join(self.m_config.scores_nonorm_dir, group)
    utils.ensure_dir(norm_dir)
    return os.path.join(norm_dir, str(model_id) + ".txt")
    
  def no_norm_result_file(self, group):
    norm_dir = self.m_config.scores_nonorm_dir
    return os.path.join(norm_dir, "scores-" + group)
    

  def zt_norm_file(self, model_id, group):
    norm_dir = os.path.join(self.m_config.scores_ztnorm_dir, group)
    utils.ensure_dir(norm_dir)
    return os.path.join(norm_dir, str(model_id) + ".txt")
    
  def zt_norm_result_file(self, group):
    norm_dir = self.m_config.scores_ztnorm_dir
    utils.ensure_dir(norm_dir)
    return os.path.join(norm_dir, "scores-" + group)
  
  def __probe_dir__(self, use_projected_dir):
    return self.m_config.projected_dir if use_projected_dir else self.m_config.features_dir
  
  def probe_files(self, group, use_projected_dir):
    """Returns the probe files used to compute the raw scores"""
    return self.m_db.objects(directory=self.__probe_dir__(use_projected_dir), extension=self.m_config.default_extension, groups=group, protocol=self.m_config.protocol, purposes="probe")
    
  def zprobe_files(self, group, use_projected_dir):
    """Returns the probe files used to compute the Z-Norm"""
    return self.m_db.zobjects(directory=self.__probe_dir__(use_projected_dir), extension=self.m_config.default_extension, protocol=self.m_config.protocol, groups=group)

  def probe_files_for_model(self, model_id, group, use_projected_dir):
    """Returns the probe files used to compute the raw scores"""
    return self.m_db.objects(directory=self.__probe_dir__(use_projected_dir), extension=self.m_config.default_extension, groups=group, protocol=self.m_config.protocol, purposes="probe", model_ids=(model_id,))
    
  def zprobe_files_for_model(self, model_id, group, use_projected_dir):
    """Returns the probe files used to compute the Z-Norm"""
    return self.m_db.zobjects(directory=self.__probe_dir__(use_projected_dir), extension=self.m_config.default_extension, protocol=self.m_config.protocol, groups=group, model_ids=(model_id,))

