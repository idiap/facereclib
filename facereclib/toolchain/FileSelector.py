#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import os
from .. import utils
import bob

class FileSelector:
  """This class provides shortcuts for selecting different files for different stages of the verification process"""

  def __init__(self,
               database,
               preprocessed_directory,
               extractor_file,
               features_directory,
               projector_file,
               projected_directory,
               enroller_file,
               model_directories,
               score_directories,
               zt_score_directories = None,
               default_extension = '.hdf5'
               ):

    """Initialize the file selector object with the current configuration"""
    self.m_database = database
    self.preprocessed_directory = preprocessed_directory
    self.extractor_file = extractor_file
    self.features_directory = features_directory
    self.projector_file = projector_file
    self.projected_directory = projected_directory
    self.enroller_file = enroller_file
    self.model_directories = model_directories
    self.score_directories = score_directories
    self.zt_score_directories = zt_score_directories
    self.default_extension = default_extension


  def make_path(self, files, directory = None, extension = None):
    """Generates a full path for the given dictionary (or dictionary of dictionary) of files."""
    if not directory:
      directory = ""
    if not extension:
      extension = ""

    # iterate over the files
    for k in files:
      if isinstance(files[k], dict):
        # dict of dict -> iterate once more
        for k2 in files[k]:
          # add directory and extension
          files[k][k2] = os.path.join(directory, files[k][k2] + extension)
      elif isinstance(files[k], tuple):
        # add directory and extension
        files[k] = (os.path.join(directory, files[k][0] + extension), files[k][1], files[k][2], files[k][3], files[k][4])
      else:
        # add directory and extension
        files[k] = os.path.join(directory, files[k] + extension)

    return files


  ### List of files that will be used for all files
  def original_image_list(self):
    """Returns the list of original images that can be used for image preprocessing"""
    files = self.m_database.all_files()
    return self.make_path(files, self.m_database.original_directory, self.m_database.original_extension)

  def annotation_list(self):
    """Returns the list of annotation files, if any (else None)"""
    if self.m_database.annotation_directory is None:
      return None
    files = self.m_database.all_files()
    files = self.make_path(files, self.m_database.annotation_directory, self.m_database.annotation_extension)
    annotations = {}
    for k in files:
      annotations[k] = utils.read_annotations(files[k], self.m_database.annotation_type)
    return annotations

  def preprocessed_image_list(self):
    """Returns the list of preprocessed images and assures that the normalized image path is existing"""
    files = self.m_database.all_files()
    return self.make_path(files, self.preprocessed_directory, self.default_extension)

  def feature_list(self):
    """Returns the list of features and assures that the feature path is existing"""
    files = self.m_database.all_files()
    return self.make_path(files, self.features_directory, self.default_extension)

  def projected_list(self):
    """Returns the list of projected features and assures that the projected feature path is existing"""
    files = self.m_database.all_files()
    return self.make_path(files, self.projected_directory, self.default_extension)



  ### Training lists
  def training_list(self, dir_type, step, sort_by_client = False):
    """Returns the list of features that should be used for projector training"""
    if dir_type == 'preprocessed':
      cur_dir = self.preprocessed_directory
    elif dir_type == 'features':
      cur_dir = self.features_directory
    elif dir_type == 'projected':
      cur_dir = self.projected_directory

    files = self.m_database.training_files(step, sort_by_client)
    return self.make_path(files, cur_dir, self.default_extension)


  ### Enrollment and models
  def model_ids(self, group):
    """Returns the sorted list of model ids from the given group"""
    return sorted(self.m_database.model_ids(group = group))

  def enroll_files(self, model_id, group, dir_type):
    """Returns the list of model features used for enrollment of the given model_id from the given group"""
    if dir_type == 'features':
      cur_dir = self.features_directory
    elif dir_type == 'projected':
      cur_dir = self.projected_directory

    files = self.m_database.enroll_files(group = group, model_id = model_id)
    return self.make_path(files, cur_dir, self.default_extension)

  def model_file(self, model_id, group):
    """Returns the file of the model and assures that the directory exists"""
    model_file = os.path.join(self.model_directories[0], group, str(model_id) + self.default_extension)
    return model_file

  def probe_objects(self, group, dir_type):
    """Returns the probe files used to compute the raw scores"""
    if dir_type == 'features':
      cur_dir = self.features_directory
    elif dir_type == 'projected':
      cur_dir = self.projected_directory

    # get the probe files for all models
    files = self.m_database.probe_objects(group = group)
    return self.make_path(files, cur_dir, self.default_extension)

  def probe_objects_for_model(self, model_id, group, dir_type):
    """Returns the probe files used to compute the raw scores"""
    if dir_type == 'features':
      cur_dir = self.features_directory
    elif dir_type == 'projected':
      cur_dir = self.projected_directory

    # get the probe files for the specific model
    files = self.m_database.probe_objects(model_id = model_id, group = group)
    return self.make_path(files, cur_dir, self.default_extension)


  def tmodel_ids(self, group):
    """Returns the sorted list of T-Norm-model ids from the given group"""
    return sorted(self.m_db.tmodels(protocol=self.m_config.protocol, groups=group, **self.__options__('models_options')))

  def tenroll_files(self, model_id, group, use_projected_dir):
    """Returns the list of T-model features used for enrollment of the given model_id from the given group"""
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
    norm_dir = os.path.join(self.score_directories[0], group)
    utils.ensure_dir(norm_dir)
    return os.path.join(norm_dir, str(model_id) + ".txt")

  def no_norm_result_file(self, group):
    norm_dir = self.score_directories[0]
    return os.path.join(norm_dir, "scores-" + group)


  def zt_norm_file(self, model_id, group):
    norm_dir = os.path.join(self.score_directories[1], group)
    utils.ensure_dir(norm_dir)
    return os.path.join(norm_dir, str(model_id) + ".txt")

  def zt_norm_result_file(self, group):
    norm_dir = self.score_directories[1]
    utils.ensure_dir(norm_dir)
    return os.path.join(norm_dir, "scores-" + group)

  def __probe_dir__(self, use_projected_dir):
    return self.m_config.projected_dir if use_projected_dir else self.m_config.features_dir

  def zprobe_files(self, group, use_projected_dir):
    """Returns the probe files used to compute the Z-Norm"""
    return self.m_db.zobjects(directory=self.__probe_dir__(use_projected_dir), extension=self.m_config.default_extension, protocol=self.m_config.protocol, groups=group)

  def zprobe_files_for_model(self, model_id, group, use_projected_dir):
    """Returns the probe files used to compute the Z-Norm"""
    return self.m_db.zobjects(directory=self.__probe_dir__(use_projected_dir), extension=self.m_config.default_extension, protocol=self.m_config.protocol, groups=group, model_ids=(model_id,), **self.__options__('models_options'))

