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
import os
import facereclib
from nose.plugins.skip import SkipTest

import pkg_resources

class DatabaseTest(unittest.TestCase):

  def config(self, resource):
    try:
      return facereclib.utils.tests.configuration_file(resource, 'database', 'databases')
    except Exception as e:
      raise SkipTest("The resource for database '%s' could not be loaded; probably you didn't define the 'bob.db.%s' in your *buildout.cfg*. Here is the import error: '%s'" % (resource, resource, e))

  def check_database(self, database, groups = ('dev',), protocol = None):
    if protocol: database.protocol = protocol
    self.assertTrue(len(database.all_files()) > 0)
    self.assertTrue(len(database.training_files('train_extractor')) > 0)
    self.assertTrue(len(database.training_files('train_enroller', arrange_by_client = True)) > 0)

    for group in groups:
      model_ids = database.model_ids(group)
      self.assertTrue(len(model_ids) > 0)
      self.assertTrue(database.client_id_from_model_id(model_ids[0]) != None)
      self.assertTrue(len(database.enroll_files(model_ids[0], group)) > 0)
      self.assertTrue(len(database.probe_files(model_ids[0], group)) > 0)

  def check_database_zt(self, database, groups = ('dev', 'eval'), protocol = None):
    if protocol: database.protocol = protocol
    self.check_database(database, groups)

    for group in groups:
      t_model_ids = database.t_model_ids(group)
      self.assertTrue(len(t_model_ids) > 0)
      self.assertTrue(database.client_id_from_model_id(t_model_ids[0]) is not None)
      self.assertTrue(len(database.t_enroll_files(t_model_ids[0], group)) > 0)
      self.assertTrue(len(database.z_probe_files(group)) > 0)

  def check_annotations(self, database):
    for file in database.all_files():
      annotations = database.annotations(file)
      if annotations is not None:
        self.assertTrue('reye' in annotations and 'leye' in annotations)


  def test01_atnt(self):
    self.check_database(self.config('atnt'))


  def test02_banca(self):
    try:
      self.check_database_zt(self.config('banca'))
      self.check_database_zt(self.config('banca_twothirds'))
      self.check_annotations(self.config('banca'))
    except IOError as e:
      raise SkipTest("The database could not queried; probably the db.sql3 file is missing. Here is the import error: '%s'" % e)


  def test03_xm2vts(self):
    try:
      self.check_database(self.config('xm2vts'), groups=('dev', 'eval'))
      self.check_database(self.config('xm2vts'), groups=('dev', 'eval'), protocol = 'darkened-lp1')
      self.check_annotations(self.config('xm2vts'))
    except IOError as e:
      raise SkipTest("The database could not queried; probably the db.sql3 file is missing. Here is the import error: '%s'" % e)


  def test04_scface(self):
    try:
      self.check_database_zt(self.config('scface'))
      self.check_annotations(self.config('scface'))
    except IOError as e:
      raise SkipTest("The database could not queried; probably the db.sql3 file is missing. Here is the import error: '%s'" % e)


  def test05_mobio(self):
    try:
      self.check_database_zt(self.config('mobio'))
      self.check_database_zt(self.config('mobio'), protocol = 'female')
      self.check_database_zt(self.config('mobio_male'))
      self.check_database_zt(self.config('mobio_female'))
      self.check_annotations(self.config('mobio'))
    except IOError as e:
      raise SkipTest("The database could not queried; probably the db.sql3 file is missing. Here is the import error: '%s'" % e)


  def test06_multipie(self):
    raise SkipTest('This test definitely takes too much time. If you want the tests enabled, please disable this exception temporarily.')
    try:
      self.check_database_zt(self.config('multipie'))
      self.check_database_zt(self.config('multipie_P'))
      self.check_database_zt(self.config('multipie_P110'))
      self.check_annotations(self.config('multipie'))
    except IOError as e:
      raise SkipTest("The database could not queried; probably the db.sql3 file is missing. Here is the import error: '%s'" % e)


  def test07_lfw(self):
    try:
      self.check_database(self.config('lfw'))
      self.check_database(self.config('lfw'), protocol = 'fold1')
      self.check_database(self.config('lfw_unrestricted'))
    except IOError as e:
      raise SkipTest("The database could not queried; probably the db.sql3 file is missing. Here is the import error: '%s'" % e)


  def test08_arface(self):
    try:
      self.check_database(self.config('arface'), groups=('dev', 'eval'))
      self.check_annotations(self.config('arface'))
    except IOError as e:
      raise SkipTest("The database could not queried; probably the db.sql3 file is missing. Here is the import error: '%s'" % e)


  def test09_gbu(self):
    try:
      self.check_database(self.config('gbu'))
      self.check_annotations(self.config('gbu'))
    except IOError as e:
      raise SkipTest("The database could not queried; probably the db.sql3 file is missing. Here is the import error: '%s'" % e)


  def test10_frgc(self):
    self.check_database(self.config('frgc'))
    self.check_database(self.config('frgc'), protocol = '2.0.2')
    self.check_annotations(self.config('frgc'))


  def test11_caspeal(self):
    try:
      self.check_database(self.config('caspeal'))
      self.check_database(self.config('caspeal'), protocol = 'aging')
      self.check_annotations(self.config('caspeal'))
    except IOError as e:
      raise SkipTest("The database could not queried; probably the db.sql3 file is missing. Here is the import error: '%s'" % e)


  def test20_verification_filelist(self):
    try:
      db1 = facereclib.utils.resources.load_resource(pkg_resources.resource_filename('facereclib.tests', os.path.join('scripts', 'atnt_Test.py')), 'database')
    except Exception as e:
      raise SkipTest("This test is skipped since the atnt database is not available.")
    try:
      db2 = facereclib.utils.resources.load_resource(pkg_resources.resource_filename('facereclib.tests', os.path.join('databases', 'atnt_fl', 'atnt_fl_database.py')), 'database')
    except Exception as e:
      raise SkipTest("This test is skipped since the verification.filelist database is not available.")
    # The test of the verification.filelist database is a bit different.
    # here, we test the output of two different ways of querying the AT&T database
    # where actually both ways are uncommon...

    # assure that different kind of queries result in the same file lists
    self.assertEqual(set([str(id) for id in db1.model_ids()]), set(db2.model_ids()))
    self.assertEqual(set([str(id) for id in db1.t_model_ids()]), set(db2.t_model_ids()))

    def check_files(f1, f2):
      self.assertEqual(set([file.path for file in f1]), set([file.path for file in f2]))

    check_files(db1.all_files(), db2.all_files())
    check_files(db1.training_files('train_extractor'), db2.training_files('train_extractor'))
    check_files(db1.enroll_files(model_id=22), db2.enroll_files(model_id='22'))
    check_files(db1.probe_files(model_id=22), db2.probe_files(model_id='22'))

    check_files(db1.t_enroll_files(model_id=22), db2.t_enroll_files(model_id='22'))
    check_files(db1.z_probe_files(), db2.z_probe_files())

    f1 = db1.all_files()[0]
    f2 = [f for f in db2.all_files() if f.path == f1.path][0]

    self.assertEqual(f1.make_path(directory='xx', extension='.yy'), f2.make_path(directory='xx', extension='.yy'))

    m1 = sorted([str(id) for id in db1.model_ids()])[0]
    m2 = sorted([str(id) for id in db2.model_ids()])[0]
    self.assertEqual(str(db1.client_id_from_model_id(m1)), db2.client_id_from_model_id(m2))

