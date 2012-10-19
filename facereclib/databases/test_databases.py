#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu May 24 10:41:42 CEST 2012
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


import unittest
import imp
import os
from nose.plugins.skip import SkipTest

class DatabaseTest(unittest.TestCase):

  def config(self, file):
    return imp.load_source('config', os.path.join('config', 'database', file)).database

  def check_database(self, database, groups = ('dev',)):
    self.assertTrue(len(database.all_files()) > 0)
    self.assertTrue(len(database.training_files('train_extractor')) > 0)
    self.assertTrue(len(database.training_files('train_enroller', arrange_by_client = True)) > 0)

    for group in groups:
      model_ids = database.model_ids(group)
      self.assertTrue(len(model_ids) > 0)
      self.assertTrue(database.client_id_from_model_id(model_ids[0]) != None)
      self.assertTrue(len(database.enroll_files(model_ids[0], group)) > 0)
      self.assertTrue(len(database.probe_files(model_ids[0], group)) > 0)

  def check_database_zt(self, database, groups = ('dev', 'eval')):
    self.check_database(database, groups)

    for group in groups:
      t_model_ids = database.t_model_ids(group)
      self.assertTrue(len(t_model_ids) > 0)
      self.assertTrue(database.client_id_from_model_id(t_model_ids[0]) != None)
      self.assertTrue(len(database.t_enroll_files(t_model_ids[0], group)) > 0)
      self.assertTrue(len(database.z_probe_files(group)) > 0)


  def test01_atnt(self):
    self.check_database(self.config('atnt_Default.py'))


  def test02_banca(self):
    self.check_database_zt(self.config('banca_P.py'))
    self.check_database_zt(self.config('banca_Ua_twothirds.py'))
    self.check_database_zt(self.config('banca_Ua_twothirds_video.py'))


  def test03_xm2vts(self):
    self.check_database(self.config('xm2vts_lp1.py'), groups=('dev', 'eval'))
    self.check_database(self.config('xm2vts_darkened.py'), groups=('dev', 'eval'))


  def test04_scface(self):
    self.check_database_zt(self.config('scface_combined.py'))


  def test05_mobio(self):
    self.check_database_zt(self.config('mobio_male.py'))
    self.check_database_zt(self.config('mobio_female.py'))


  def test06_multipie(self):
    self.check_database_zt(self.config('multipie_U.py'))
    self.check_database_zt(self.config('multipie_P.py'))
    self.check_database_zt(self.config('multipie_left_profile.py'))


  def test07_lfw(self):
    self.check_database(self.config('lfw_view1.py'))
    self.check_database(self.config('lfw_view1_unrestricted.py'))

