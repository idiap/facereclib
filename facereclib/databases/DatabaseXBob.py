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

class DatabaseXBob (Database):
  def __init__(self,
               database,
               name, # The name of the database; will be used as part of the directory structure
               image_directory, # the directory where to read the images from
               image_extension, # the default file extension of the original images
               annotation_directory = None, # The directory, where the annotations are found, if any
               annotation_extension = '.pos', # The extension of the annotation files
               annotation_type = None, # The way the annotations are written in the annotation files
               protocol = 'Default',

               all_files_options = {}, # additional options for the database query that can be used to extract all files
               extractor_training_options = {}, # additional options for the database query that can be used to extract the training files for the extractor training
               projector_training_options = {}, # additional options for the database query that can be used to extract the training files for the extractor training
               enroller_training_options = {} # additional options for the database query that can be used to extract the training files for the extractor training
               ):

    Database.__init__(self,
                      name = name,
                      original_directory = image_directory,
                      original_extension = image_extension,
                      annotation_directory = annotation_directory,
                      annotation_extension = annotation_extension,
                      annotation_type = annotation_type,
                      protocol = protocol)

    self.m_database = database

    self.all_files_options = all_files_options
    self.extractor_training_options = extractor_training_options
    self.projector_training_options = projector_training_options
    self.enroller_training_options = enroller_training_options


  def all_files(self):
    files = self.m_database.objects(protocol = self.protocol, **self.all_files_options)
    return self.sort(files)


  def training_files(self, step, arrange_by_client = False):
    if step == 'train_extractor':
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


  def model_ids(self, group = 'dev'):
    if hasattr(self.m_database, 'model_ids'):
      return sorted(self.m_database.model_ids(protocol = self.protocol, groups = group))
    else:
      return sorted([model.id for model in self.m_database.models(protocol = self.protocol, groups = group)])

  def client_id_from_model_id(self, model_id):
    if hasattr(self.m_database, 'get_client_id_from_model_id'):
      return self.m_database.get_client_id_from_model_id(model_id)
    else:
      return model_id

  def enroll_files(self, model_id, group = 'dev'):
    files = self.m_database.objects(protocol = self.protocol, groups = group, model_ids = (model_id,), purposes = 'enrol')
    return self.sort(files)


  def probe_files(self, model_id = None, group = 'dev'):
    if model_id:
      files = self.m_database.objects(protocol = self.protocol, groups = group, model_ids = (model_id,), purposes = 'probe')
    else:
      files = self.m_database.objects(protocol = self.protocol, groups = group, purposes = 'probe')
    return self.sort(files)



class DatabaseXBobZT (DatabaseXBob, DatabaseZT):

  def __init__(self,
               database,
               name, # The name of the database; will be used as part of the directory structure
               image_directory, # the directory where to read the images from
               image_extension, # the default file extension of the original images
               annotation_directory = None, # The directory, where the annotations are found, if any
               annotation_extension = '.pos', # The extension of the annotation files
               annotation_type = None, # The way the annotations are written in the annotation files
               protocol = 'Default',

               all_files_options = {}, # additional options for the database query that can be used to extract all files
               extractor_training_options = {}, # additional options for the database query that can be used to extract the training files for the extractor training
               projector_training_options = {}, # additional options for the database query that can be used to extract the training files for the extractor training
               enroller_training_options = {} # additional options for the database query that can be used to extract the training files for the extractor training
               ):

    DatabaseXBob.__init__(self,
                          database = database,
                          name = name,
                          image_directory = image_directory,
                          image_extension = image_extension,
                          annotation_directory = annotation_directory,
                          annotation_extension = annotation_extension,
                          annotation_type = annotation_type,
                          protocol = protocol,
                          all_files_options = all_files_options,
                          extractor_training_options = extractor_training_options,
                          projector_training_options = projector_training_options,
                          enroller_training_options = enroller_training_options)


  def t_model_ids(self, group = 'dev'):
    return sorted([model.id for model in self.m_database.tmodels(protocol = self.protocol, groups = group)])


  def t_enroll_files(self, model_id, group = 'dev'):
    files = self.m_database.tobjects(protocol = self.protocol, groups = group, model_ids = (model_id,))
    return self.sort(files)


  def z_probe_files(self, group = 'dev'):
    files = self.m_database.zobjects(protocol = self.protocol, groups = group)
    return self.sort(files)

