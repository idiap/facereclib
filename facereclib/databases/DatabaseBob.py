#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Wed Oct  3 10:31:51 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from .Database import Database, DatabaseZT

class DatabaseBob (Database):
  """This class can be used whenever you have a database that follows the default Bob database interface."""

  def __init__(
      self,
      database,  # The bob database that is used
      all_files_options = {}, # additional options for the database query that can be used to extract all files
      extractor_training_options = {}, # additional options for the database query that can be used to extract the training files for the extractor training
      projector_training_options = {}, # additional options for the database query that can be used to extract the training files for the extractor training
      enroller_training_options = {},  # additional options for the database query that can be used to extract the training files for the extractor training
      check_original_files_for_existence = False,
      **kwargs  # The default parameters of the base class
  ):
    """
    Parameters of the constructor of this database:

    database
      the bob.db.___ database that provides the actual interface

    image_directory
      The directory where the original images are stored.

    image_extension
      The file extension of the original images.

    all_files_options
      Options passed to the database query used to retrieve all data.

    extractor_training_options
      Options passed to the database query used to retrieve the images for the extractor training.

    projector_training_options
      Options passed to the database query used to retrieve the images for the projector training.

    enroller_training_options
      Options passed to the database query used to retrieve the images for the enroller training.

    check_original_files_for_existence
      Enables the test for the original data files when querying the database.

    kwargs
      The arguments of the base class
    """

    Database.__init__(
        self,
        **kwargs
    )

    self.m_database = database
    self.original_directory = database.original_directory

    self.all_files_options = all_files_options
    self.extractor_training_options = extractor_training_options
    self.projector_training_options = projector_training_options
    self.enroller_training_options = enroller_training_options
    self.check_existence = check_original_files_for_existence

    self._kwargs = kwargs


  def __str__(self):
    """This function returns a string containing all parameters of this class (and its derived class)."""
    params = ", ".join(["%s=%s" % (key, value) for key, value in self._kwargs.items()])
    params += ", original_directory=%s, original_extension=%s" % (self.original_directory, self.original_extension)
    if self.all_files_options: params += ", all_files_options=%s"%self.all_files_options
    if self.extractor_training_options: params += ", extractor_training_options=%s"%self.extractor_training_options
    if self.projector_training_options: params += ", projector_training_options=%s"%self.projector_training_options
    if self.enroller_training_options: params += ", enroller_training_options=%s"%self.enroller_training_options

    return "%s(%s)" % (str(self.__class__), params)


  def uses_probe_file_sets(self):
    """Defines if, for the current protocol, the database uses several probe files to generate a score."""
    return self.protocol != 'None' and self.m_database.provides_file_set_for_protocol(self.protocol)


  def all_files(self, groups = None):
    """Returns all File objects of the database for the current protocol. If the current protocol is 'None' (a string), None (NoneType) will be used instead"""
    files = self.m_database.objects(protocol = self.protocol if self.protocol != 'None' else None, groups = groups, **self.all_files_options)
    return self.sort(files)


  def training_files(self, step = None, arrange_by_client = False):
    """Returns all training File objects of the database for the current protocol."""
    if step is None:
      training_options = self.all_files_options
    elif step == 'train_extractor':
      training_options = self.extractor_training_options
    elif step == 'train_projector':
      training_options = self.projector_training_options
    elif step == 'train_enroller':
      training_options = self.enroller_training_options
    else:
      raise ValueError("The given step '%s' must be one of ('train_extractor', 'train_projector', 'train_enroller')" % step)

    files = self.sort(self.m_database.objects(protocol = self.protocol, groups = 'world', **training_options))
    if arrange_by_client:
      return self.arrange_by_client(files)
    else:
      return files

  def test_files(self, groups = ['dev']):
    """Returns the test files (i.e., enrollment and probe files) for the given groups."""
    return self.sort(self.m_database.test_files(protocol = self.protocol, groups = groups, **self.all_files_options))

  def model_ids(self, group = 'dev'):
    """Returns the model ids for the given group and the current protocol."""
    if hasattr(self.m_database, 'model_ids'):
      return sorted(self.m_database.model_ids(protocol = self.protocol, groups = group))
    else:
      return sorted([model.id for model in self.m_database.models(protocol = self.protocol, groups = group)])


  def client_id_from_model_id(self, model_id, group = 'dev'):
    """Returns the client id for the given model id."""
    if hasattr(self.m_database, 'get_client_id_from_model_id'):
      return self.m_database.get_client_id_from_model_id(model_id)
    else:
      return model_id


  def enroll_files(self, model_id, group = 'dev'):
    """Returns the list of enrollment File objects for the given model id."""
    files = self.m_database.objects(protocol = self.protocol, groups = group, model_ids = (model_id,), purposes = 'enroll', **self.all_files_options)
    return self.sort(files)


  def probe_files(self, model_id = None, group = 'dev'):
    """Returns the list of probe File objects (for the given model id, if given)."""
    if model_id:
      files = self.m_database.objects(protocol = self.protocol, groups = group, model_ids = (model_id,), purposes = 'probe', **self.all_files_options)
    else:
      files = self.m_database.objects(protocol = self.protocol, groups = group, purposes = 'probe', **self.all_files_options)
    return self.sort(files)


  def probe_file_sets(self, model_id = None, group = 'dev'):
    """Returns the list of probe File objects (for the given model id, if given)."""
    if model_id:
      file_sets = self.m_database.object_sets(protocol = self.protocol, groups = group, model_ids = (model_id,), purposes = 'probe', **self.all_files_options)
    else:
      file_sets = self.m_database.object_sets(protocol = self.protocol, groups = group, purposes = 'probe', **self.all_files_options)
    return self.sort(file_sets)


  def annotations(self, file):
    """Returns the annotations for the given File object, if available."""
    return self.m_database.annotations(file)


  def original_file_names(self, files):
    """Returns the full path of the original data of the given File objects."""
    return self.m_database.original_file_names(files, self.check_existence)



class DatabaseBobZT (DatabaseBob, DatabaseZT):
  """This class can be used whenever you have a database that follows the default Bob database interface defining file lists for ZT score normalization."""

  def __init__(
      self,
      z_probe_options = {}, # Limit the z-probes
      **kwargs
  ):
    # call base class constructor, passing all the parameters to it
    DatabaseBob.__init__(self, z_probe_options = z_probe_options, **kwargs)

    self.m_z_probe_options = z_probe_options


  def all_files(self, groups = ['dev']):
    """Returns all File objects of the database for the current protocol. If the current protocol is 'None' (a string), None (NoneType) will be used instead"""
    files = self.m_database.objects(protocol = self.protocol if self.protocol != 'None' else None, groups = groups, **self.all_files_options)

    # add all files that belong to the ZT-norm
    for group in groups:
      if group == 'world': continue
      files += self.m_database.tobjects(protocol = self.protocol if self.protocol != 'None' else None, groups = group, model_ids = None)
      files += self.m_database.zobjects(protocol = self.protocol if self.protocol != 'None' else None, groups = group, **self.m_z_probe_options)
    return self.sort(files)


  def t_model_ids(self, group = 'dev'):
    """Returns the T-Norm model ids for the given group and the current protocol."""
    if hasattr(self.m_database, 'tmodel_ids'):
      return sorted(self.m_database.tmodel_ids(protocol = self.protocol, groups = group))
    else:
      return sorted([model.id for model in self.m_database.tmodels(protocol = self.protocol, groups = group)])


  def t_enroll_files(self, model_id, group = 'dev'):
    """Returns the list of enrollment File objects for the given T-Norm model id."""
    files = self.m_database.tobjects(protocol = self.protocol, groups = group, model_ids = (model_id,))
    return self.sort(files)


  def z_probe_files(self, group = 'dev'):
    """Returns the list of Z-probe File objects."""
    files = self.m_database.zobjects(protocol = self.protocol, groups = group, **self.m_z_probe_options)
    return self.sort(files)


  def z_probe_file_sets(self, group = 'dev'):
    """Returns the list of Z-probe Fileset objects."""
    file_sets = self.m_database.zobject_sets(protocol = self.protocol, groups = group, **self.m_z_probe_options)
    return self.sort(file_sets)
