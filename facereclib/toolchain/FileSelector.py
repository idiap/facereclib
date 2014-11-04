#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import os
from .. import utils

class FileSelector:
  """This class provides shortcuts for selecting different files for different stages of the verification process"""

  def __init__(
        self,
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

    """Initialize the file selector object with the current configuration."""
    self.m_database = database
    self.original_directory = database.original_directory
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


  def uses_probe_file_sets(self):
    """Returns true if the given protocol enables several probe files for scoring."""
    return self.m_database.uses_probe_file_sets()

  def get_paths(self, files, directory_type = None):
    """Returns the list of file names for the given list of File objects."""
    if directory_type == 'preprocessed':
      directory = self.preprocessed_directory
    elif directory_type == 'features':
      directory = self.features_directory
    elif directory_type == 'projected':
      directory = self.projected_directory
    else:
      raise ValueError("The given directory type '%s' is not supported." % directory_type)

    return self.m_database.file_names(files, directory, self.default_extension)


  ### List of files that will be used for all files
  def original_data_list(self, groups = None):
    """Returns the list of original data that can be used for preprocessing."""
    return self.m_database.original_file_names(self.m_database.all_files(groups=groups))

  def annotation_list(self, groups = None):
    """Returns the list of annotations objects."""
    return self.m_database.all_files(groups=groups)

  def get_annotations(self, annotation_file):
    """Returns the annotations of the given file."""
    return self.m_database.annotations(annotation_file)

  def preprocessed_data_list(self, groups = None):
    """Returns the list of preprocessed data files."""
    return self.get_paths(self.m_database.all_files(groups=groups), "preprocessed")

  def feature_list(self, groups = None):
    """Returns the list of extracted feature files."""
    return self.get_paths(self.m_database.all_files(groups=groups), "features")

  def projected_list(self, groups = None):
    """Returns the list of projected feature files."""
    return self.get_paths(self.m_database.all_files(groups=groups), "projected")


  ### Training lists
  def training_list(self, directory_type, step, arrange_by_client = False):
    """Returns the list of features that should be used for projector training.
    The directory_type might be any of 'preprocessed', 'features', or 'projected'.
    The step might by any of 'train_extractor', 'train_projector', or 'train_enroller'.
    If arrange_by_client is enabled, a list of lists (one list for each client) is returned."""
    files = self.m_database.training_files(step, arrange_by_client)
    if arrange_by_client:
      return [self.get_paths(files[client], directory_type) for client in range(len(files))]
    else:
      return self.get_paths(files, directory_type)


  ### Enrollment and models
  def client_id(self, model_id, group, is_t_model_id = False):
    """Returns the id of the client for the given model id or T-norm model id."""
    if is_t_model_id:
      return self.m_database.client_id_from_t_model_id(model_id, group = group)
    else:
      return self.m_database.client_id_from_model_id(model_id, group = group)

  def model_ids(self, group):
    """Returns the sorted list of model ids from the given group."""
    return sorted(self.m_database.model_ids(group = group))

  def enroll_files(self, model_id, group, directory_type):
    """Returns the list of model feature files used for enrollment of the model with the given model_id from the given group.
    The directory_type might be features' or 'projected'."""
    files = self.m_database.enroll_files(group = group, model_id = model_id)
    return self.get_paths(files, directory_type)

  def model_file(self, model_id, group):
    """Returns the file of the model with the given model id."""
    return os.path.join(self.model_directories[0], group, str(model_id) + self.default_extension)

  def probe_objects(self, group):
    """Returns the probe File objects used to compute the raw scores."""
    # get the probe files for all models
    if self.uses_probe_file_sets():
      return self.m_database.probe_file_sets(group = group)
    else:
      return self.m_database.probe_files(group = group)

  def probe_objects_for_model(self, model_id, group):
    """Returns the probe File objects used to compute the raw scores for the given model id.
    This is actually a sub-set of all probe_objects()."""
    # get the probe files for the specific model
    if self.uses_probe_file_sets():
      return self.m_database.probe_file_sets(model_id = model_id, group = group)
    else:
      return self.m_database.probe_files(model_id = model_id, group = group)


  def t_model_ids(self, group):
    """Returns the sorted list of T-Norm-model ids from the given group."""
    return sorted(self.m_database.t_model_ids(group = group))

  def t_enroll_files(self, model_id, group, directory_type):
    """Returns the list of T-norm model files used for enrollment of the given model_id from the given group."""
    files = self.m_database.t_enroll_files(group = group, model_id = model_id)
    return self.get_paths(files, directory_type)

  def t_model_file(self, model_id, group):
    """Returns the file of the T-Norm-model with the given model id."""
    return os.path.join(self.model_directories[1], group, str(model_id) + self.default_extension)

  def z_probe_objects(self, group):
    """Returns the probe File objects used to compute the Z-Norm."""
    # get the probe files for all models
    if self.uses_probe_file_sets():
      return self.m_database.z_probe_file_sets(group = group)
    else:
      return self.m_database.z_probe_files(group = group)


  ### ZT-Normalization
  def a_file(self, model_id, group):
    """Returns the A-file for the given model id that is used for computing ZT normalization."""
    a_dir = os.path.join(self.zt_score_directories[0], group)
    utils.ensure_dir(a_dir)
    return os.path.join(a_dir, str(model_id) + self.default_extension)

  def b_file(self, model_id, group):
    """Returns the B-file for the given model id that is used for computing ZT normalization."""
    b_dir = os.path.join(self.zt_score_directories[1], group)
    utils.ensure_dir(b_dir)
    return os.path.join(b_dir, str(model_id) + self.default_extension)

  def c_file(self, t_model_id, group):
    """Returns the C-file for the given T-model id that is used for computing ZT normalization."""
    c_dir = os.path.join(self.zt_score_directories[2], group)
    utils.ensure_dir(c_dir)
    return os.path.join(c_dir, "TM" + str(t_model_id) + self.default_extension)

  def c_file_for_model(self, model_id, group):
    """Returns the C-file for the given model id that is used for computing ZT normalization."""
    c_dir = os.path.join(self.zt_score_directories[2], group)
    utils.ensure_dir(c_dir)
    return os.path.join(c_dir, str(model_id) + self.default_extension)

  def d_file(self, t_model_id, group):
    """Returns the D-file for the given T-model id that is used for computing ZT normalization."""
    d_dir = os.path.join(self.zt_score_directories[3], group)
    utils.ensure_dir(d_dir)
    return os.path.join(d_dir, str(t_model_id) + self.default_extension)

  def d_matrix_file(self, group):
    """Returns the D-file for storing all scores for pairs of T-models and Z-probes."""
    d_dir = os.path.join(self.zt_score_directories[3], group)
    utils.ensure_dir(d_dir)
    return os.path.join(d_dir, "D" + self.default_extension)

  def d_same_value_file(self, t_model_id, group):
    """Returns the specific D-file for storing which pairs of the given T-model id and all Z-probes are intrapersonal or extrapersonal."""
    d_dir = os.path.join(self.zt_score_directories[4], group)
    utils.ensure_dir(d_dir)
    return os.path.join(d_dir, str(t_model_id) + self.default_extension)

  def d_same_value_matrix_file(self, group):
    """Returns the specific D-file for storing which pairs of T-models and Z-probes are intrapersonal or extrapersonal."""
    d_dir = os.path.join(self.zt_score_directories[4], group)
    utils.ensure_dir(d_dir)
    return os.path.join(d_dir, "D_sameValue" + self.default_extension)

  def no_norm_file(self, model_id, group):
    """Returns the score text file for the given model id of the given group."""
    no_norm_dir = os.path.join(self.score_directories[0], group)
    utils.ensure_dir(no_norm_dir)
    return os.path.join(no_norm_dir, str(model_id) + ".txt")

  def no_norm_result_file(self, group):
    """Returns the resulting score text file for the given group."""
    no_norm_dir = self.score_directories[0]
    utils.ensure_dir(no_norm_dir)
    return os.path.join(no_norm_dir, "scores-" + group)


  def zt_norm_file(self, model_id, group):
    """Returns the score text file after ZT-normalization for the given model id of the given group."""
    zt_norm_dir = os.path.join(self.score_directories[1], group)
    utils.ensure_dir(zt_norm_dir)
    return os.path.join(zt_norm_dir, str(model_id) + ".txt")

  def zt_norm_result_file(self, group):
    """Returns the resulting score text file after ZT-normalization for the given group."""
    zt_norm_dir = self.score_directories[1]
    utils.ensure_dir(zt_norm_dir)
    return os.path.join(zt_norm_dir, "scores-" + group)

  def calibrated_score_file(self, group, zt_norm=False):
    """Returns the directory where calibrated scores can be found."""
    calibration_dir = self.score_directories[1 if zt_norm else 0]
    utils.ensure_dir(calibration_dir)
    return os.path.join(calibration_dir, "calibrated-" + group)

