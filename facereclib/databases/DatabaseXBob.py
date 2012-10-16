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


from .Database import Database

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

    Database.__init__(self, name, image_directory, image_extension, annotation_directory, annotation_extension, annotation_type, protocol)

    self.m_database = database

    self.all_files_options = all_files_options
    self.extractor_training_options = extractor_training_options
    self.projector_training_options = projector_training_options
    self.enroller_training_options = enroller_training_options

  def make_dict(self, objects):
    """This function generates the dictionary for the given objects.
    In this implementation, the object.id (file id) is related to the object.path.
    If you have different types of names in your objects, please overwrite this function.
    """
    res={}
    for object in objects:
      res[object.id] = object.path
    return res

  def make_object_dict(self, objects, model_id, client_id):
    """This function generates the dictionary for the given objects.
    In this implementation, the object.id (file id) is related to a list of:

    * 0: object.path -> the file name of the object
    * 1: model_id -> the model id given as parameter
    * 2: client_id -> the client id of the claimed client id attached to the model (in case of the AR database: identical to 1)
    * 3: object.client_id -> the client id that is attached to the probe
    * 4: object.path (again)

    If you have different types of names in your objects, please overwrite this function.
    """
    res={}
    for object in objects:
      res[object.id] = (object.path, model_id, client_id, object.client_id, object.path)
    return res


  def make_dict_by_client(self, objects):
    """This function generates the dictionary of dictionaries for the given objects.
    In this implementation, the object.client_id (client id), object.id (file id) and the object.path are related.
    If you have different types of names in your objects, please overwrite this function.
    """
    res={}
    for object in objects:
      if not object.real_id in res:
        res[object.real_id] = {}
      res[object.real_id][object.id] = object.path
    return res


  def all_files(self):
    files = self.m_database.objects(protocol = self.protocol, **self.all_files_options)
    return self.make_dict(files)


  def training_files(self, step, sort_by_client):
    if step == 'train_extractor':
      training_options = self.extractor_training_options
    elif step == 'train_projector':
      training_options = self.projector_training_options
    elif step == 'train_enroller':
      training_options = self.enroller_training_options
    else:
      raise ValueError("The given step '%s' must be one of ('extract', 'project', 'enroll')" % step)

    files = self.m_database.objects(protocol = self.protocol, groups = 'world', **training_options)
    return self.make_dict_by_client(files) if sort_by_client else self.make_dict(files)


  def client_ids(self, group = 'dev'):
    return [client.id for client in self.m_database.clients(protocol = self.protocol, groups = group)]

  def model_ids(self, group = 'dev'):
    return self.client_ids(group)


  def enroll_files(self, model_id, group = 'dev'):
    files = self.m_database.objects(protocol = self.protocol, groups = group, model_ids = (model_id,), purposes = 'enrol')
    return self.make_dict(files)


  def probe_files(self, model_id, group = 'dev'):
    files = self.m_database.objects(protocol = self.protocol, groups = group, model_ids = (model_id,), purposes = 'probe')
    return self.make_dict(files)

  def probe_objects(self, model_id, group = 'dev'):
    files = self.m_database.objects(protocol = self.protocol, groups = group, model_ids = (model_id,), purposes = 'probe')
    client = self.m_database.client(get_client_id_from_model_id)
    return self.make_object_dict(files, model_id, client.id)
