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

class DatabaseZT (Database):
  def __init__(self,
               name, # The name of the database; will be used as part of the directory structure
               image_directory, # the directory where to read the images from
               image_extension, # the default file extension of the original images
               annotation_directory = None, # The directory, where the annotations are found, if any
               annotation_extension = '.pos', # The extension of the annotation files
               annotation_type = None, # The way the annotations are written in the annotation files
               protocol = 'Default'
               ):

    Database.__init__(self, name, image_directory, image_extension, annotation_directory, annotation_extension, annotation_type, protocol)


  def t_model_ids(self, group = 'dev'):
    """Returns a list of T-Norm model ids for the given group"""
    raise NotImplementedError("Please implement this function in derived classes")


  def t_enroll_files(self, model_id, group = 'dev'):
    """Returns a list of T-Norm enrollment files for the given model id and the given group"""
    raise NotImplementedError("Please implement this function in derived classes")


  def z_probe_objects(self, model_id = None, group = 'dev'):
    """Returns a list of probe objects for Z-Norm in a specific format that should be compared with the model belonging to the given model id of the specified group"""
    raise NotImplementedError("Please implement this function in derived classes")

