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

import os

class Database:
  def __init__(self, name, original_directory, original_extension, annotation_directory = None, annotation_extension = '.pos', annotation_type = None, protocol = 'Default'):
    self.name = name
    self.original_directory = original_directory
    self.original_extension = original_extension
    self.annotation_directory = annotation_directory
    self.annotation_extension = annotation_extension
    self.annotation_type = annotation_type
    self.protocol = protocol


  def all_files(self):
    """Returns all files of the database"""
    raise NotImplementedError("Please implement this function in derived classes")


  def training_files(self, step, sort_by_client = False):
    """Returns all training files for the given step (might be 'extract, project, enroll'), and arranges them by client, if desired"""
    raise NotImplementedError("Please implement this function in derived classes")


  def client_ids(self, group = 'dev'):
    """Returns a list of client ids for the given group"""
    raise NotImplementedError("Please implement this function in derived classes")

  def model_ids(self, group = 'dev'):
    """Returns a list of model ids for the given group"""
    raise NotImplementedError("Please implement this function in derived classes")


  def enroll_files(self, model_id, group = 'dev'):
    """Returns a list of enrollment files for the given model id and the given group"""
    raise NotImplementedError("Please implement this function in derived classes")


  def probe_files(self, model_id, group = 'dev'):
    """Returns a list of probe files that should be compared with the model belonging to the given model id of the specified group"""
    raise NotImplementedError("Please implement this function in derived classes")

  def probe_objects(self, model_id = None, group = 'dev'):
    """Returns a list of probe objects in a specific format that should be compared with the model belonging to the given model id of the specified group"""
    raise NotImplementedError("Please implement this function in derived classes")

  ### helper functions

