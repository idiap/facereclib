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
from __future__ import print_function

import bob.measure

import unittest
import os
import sys
import shutil
import tempfile
import numpy

import facereclib
from nose.plugins.skip import SkipTest

import pkg_resources

base_dir = pkg_resources.resource_filename('facereclib', 'tests')
config_dir = pkg_resources.resource_filename('facereclib', 'configurations')

class ScriptTest (unittest.TestCase):

  def __face_verify__(self, parameters, test_dir, sub_dir, ref_modifier="", score_modifier=('scores','')):
    from facereclib.script.faceverify import main
    main([sys.argv[0]] + parameters)

    # assert that the score file exists
    score_files = (os.path.join(test_dir, sub_dir, 'scores', 'Default', 'nonorm', '%s-dev%s'%score_modifier), os.path.join(test_dir, sub_dir, 'scores', 'Default', 'ztnorm', '%s-dev%s'%score_modifier))
    self.assertTrue(os.path.exists(score_files[0]))
    self.assertTrue(os.path.exists(score_files[1]))

    # also assert that the scores are still the same -- though they have no real meaning
    reference_files = (os.path.join(base_dir, 'scripts', 'scores-nonorm%s-dev'%ref_modifier), os.path.join(base_dir, 'scripts', 'scores-ztnorm%s-dev'%ref_modifier))

    for i in (0,1):
      d = []
      # read reference and new data
      for score_file in (score_files[i], reference_files[i]):
        f = bob.measure.load.open_file(score_files[i])
        d_ = []
        for line in f:
          if isinstance(line, bytes): line = line.decode('utf-8')
          d_.append(line.rstrip().split())
        d.append(numpy.array(d_))

      self.assertTrue(d[0].shape, d[1].shape)
      # assert that the data order is still correct
      self.assertTrue((d[0][:,0:3] == d[1][:, 0:3]).all())
      # assert that the values are OK
      self.assertTrue((numpy.abs(d[0][:,3].astype(float) - d[1][:,3].astype(float)) < 1e-5).all())

    shutil.rmtree(test_dir)


  def grid_available(self):
    try:
      import gridtk
    except ImportError:
      raise SkipTest("Skipping test since gridtk is not available")


  def test01_faceverify_local(self):
    test_dir = tempfile.mkdtemp(prefix='frltest_')
    # define dummy parameters
    parameters = [
        '-d', os.path.join(base_dir, 'scripts', 'atnt_Test.py'),
        '-p', os.path.join(config_dir, 'preprocessing', 'face_crop.py'),
        '-f', os.path.join(config_dir, 'features', 'eigenfaces.py'),
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '--zt-norm',
        '-b', 'test',
        '--temp-directory', test_dir,
        '--user-directory', test_dir
    ]

    print (facereclib.utils.command_line(parameters))

    self.__face_verify__(parameters, test_dir, 'test')


  def test01a_faceverify_resources(self):
    test_dir = tempfile.mkdtemp(prefix='frltest_')
    # define dummy parameters
    parameters = [
        '-d', os.path.join(base_dir, 'scripts', 'atnt_Test.py'),
        '-p', 'face-crop',
        '-f', 'eigenfaces',
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '--zt-norm',
        '-b', 'test_a',
        '--temp-directory', test_dir,
        '--user-directory', test_dir
    ]

    print (facereclib.utils.command_line(parameters))

    self.__face_verify__(parameters, test_dir, 'test_a')


  def test01b_faceverify_commandline(self):
    test_dir = tempfile.mkdtemp(prefix='frltest_')
    # define dummy parameters
    parameters = [
        '-d', os.path.join(base_dir, 'scripts', 'atnt_Test.py'),
        '-p', 'face-crop',
        '-f', 'facereclib.features.Eigenface(subspace_dimension', '=', '100)',
        '-t', 'facereclib.tools.Dummy()',
        '--zt-norm',
        '-b', 'test_b',
        '--temp-directory', test_dir,
        '--user-directory', test_dir
    ]

    print (facereclib.utils.command_line(parameters))

    self.__face_verify__(parameters, test_dir, 'test_b')


  def test01c_faceverify_parallel(self):
    self.grid_available()
    test_dir = tempfile.mkdtemp(prefix='frltest_')
    test_database = os.path.join(test_dir, "database.sql3")

    # define dummy parameters
    parameters = [
        '-d', os.path.join(base_dir, 'scripts', 'atnt_Test.py'),
        '-p', 'face-crop',
        '-f', 'facereclib.features.Eigenface(subspace_dimension', '=', '100)',
        '-t', 'facereclib.tools.Dummy()',
        '--zt-norm',
        '-b', 'test_c',
        '--temp-directory', test_dir,
        '--user-directory', test_dir,
        '-g', 'facereclib.utils.GridParameters(grid = "local", number_of_parallel_processes = 2, scheduler_sleep_time = 0.1)', '-G', test_database, '--run-local-scheduler'
    ]

    print (facereclib.utils.command_line(parameters))

    self.__face_verify__(parameters, test_dir, 'test_c')


  def test01d_faceverify_compressed(self):
    test_dir = tempfile.mkdtemp(prefix='frltest_')
    # define dummy parameters
    parameters = [
        '-d', os.path.join(base_dir, 'scripts', 'atnt_Test.py'),
        '-p', 'face-crop',
        '-f', 'facereclib.features.Eigenface(subspace_dimension', '=', '100)',
        '-t', 'facereclib.tools.Dummy()',
        '--zt-norm',
        '-b', 'test_d',
        '--temp-directory', test_dir,
        '--user-directory', test_dir,
        '--write-compressed-score-files'
    ]

    print (facereclib.utils.command_line(parameters))

    self.__face_verify__(parameters, test_dir, 'test_d', score_modifier=('scores', '.tar.bz2'))


  def test01m_faceverify_calibrate(self):
    test_dir = tempfile.mkdtemp(prefix='frltest_')
    # define dummy parameters
    parameters = [
        '-d', os.path.join(base_dir, 'scripts', 'atnt_Test.py'),
        '-p', os.path.join(config_dir, 'preprocessing', 'face_crop.py'),
        '-f', os.path.join(config_dir, 'features', 'eigenfaces.py'),
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '--zt-norm',
        '-b', 'test',
        '--temp-directory', test_dir,
        '--user-directory', test_dir,
        '--calibrate-scores'
    ]

    print (facereclib.utils.command_line(parameters))

    # check that the calibrated scores are as expected
    self.__face_verify__(parameters, test_dir, 'test', '-calibrated', score_modifier=('calibrated', ''))


  def test01x_faceverify_filelist(self):
    try:
      import bob.db.verification.filelist
    except ImportError:
      raise SkipTest("Skipping test since bob.db.verification.filelist is not available")
    test_dir = tempfile.mkdtemp(prefix='frltest_')
    # define dummy parameters
    parameters = [
        '-d', os.path.join(base_dir, 'databases', 'atnt_fl', 'atnt_fl_database.py'),
#        '--protocol', 'None',
        '-p', os.path.join(config_dir, 'preprocessing', 'face_crop.py'),
        '-f', os.path.join(config_dir, 'features', 'eigenfaces.py'),
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '--zt-norm',
        '-b', 'test_x',
        '--temp-directory', test_dir,
        '--user-directory', test_dir
    ]

    print (facereclib.utils.command_line(parameters))

    from facereclib.script.faceverify import main
    main([sys.argv[0]] + parameters)

    # assert that the score file exists
    score_files = (os.path.join(test_dir, 'test_x', 'scores', 'nonorm', 'scores-dev'), os.path.join(test_dir, 'test_x', 'scores', 'ztnorm', 'scores-dev'))
    self.assertTrue(os.path.exists(score_files[0]))
    self.assertTrue(os.path.exists(score_files[1]))

    # assert that the scores are are identical
    reference_files = (os.path.join(base_dir, 'scripts', 'scores-nonorm-dev'), os.path.join(base_dir, 'scripts', 'scores-ztnorm-dev'))

    for i in (0,1):

      a1, b1 = bob.measure.load.split_four_column(score_files[i])
      a2, b2 = bob.measure.load.split_four_column(reference_files[i])

      a1 = sorted(a1); a2 = sorted(a2); b1 = sorted(b1); b2 = sorted(b2)

      for i in range(len(a1)):
        self.assertAlmostEqual(a1[i], a2[i], 6)
      for i in range(len(b1)):
        self.assertAlmostEqual(b1[i], b2[i], 6)

    shutil.rmtree(test_dir)



  def test02_faceverify_grid(self):
    self.grid_available()
    test_dir = tempfile.mkdtemp(prefix='frltest_')
    # define dummy parameters including the dry-run
    parameters = [
        sys.argv[0],
        '-d', os.path.join(base_dir, 'scripts', 'atnt_Test.py'),
        '-p', 'face-crop',
        '-f', 'eigenfaces',
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '-g', 'grid',
        '--zt-norm',
        '--dry-run',
        '--user-directory', test_dir,
        '-b', 'dummy'
    ]

    print (facereclib.utils.command_line(parameters))

    # run the test; should not execute anything...
    from facereclib.script.faceverify import main
    main(parameters)
    shutil.rmtree(test_dir)


  def test03_faceverify_lfw_local(self):
    # try to import the lfw database
    try:
      facereclib.utils.resources.load_resource('lfw','database')
    except Exception as e:
      raise SkipTest("The resource for database 'lfw' could not be loaded; probably you didn't define the 'bob.db.lfw' in your *buildout.cfg*. Here is the import error: '%s'" % e)

    test_dir = tempfile.mkdtemp(prefix='frltest_')
    # define dummy parameters
    parameters = [
        sys.argv[0],
        '-p', 'face-crop',
        '-f', 'eigenfaces',
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '--dry-run',
        '--user-directory', test_dir,
        '-b', 'dummy'
    ]

    print (facereclib.utils.command_line(parameters))

    # run the test; should not execute anything...
    from facereclib.script.faceverify_lfw import main
    main(parameters)
    shutil.rmtree(test_dir)


  def test04_faceverify_lfw_grid(self):
    self.grid_available()
    # try to import the lfw database
    try:
      facereclib.utils.resources.load_resource('lfw','database')
    except Exception as e:
      raise SkipTest("The resource for database 'lfw' could not be loaded; probably you didn't define the 'bob.db.lfw' in your *buildout.cfg*. Here is the import error: '%s'" % e)

    test_dir = tempfile.mkdtemp(prefix='frltest_')
    # define dummy parameters
    parameters = [
        sys.argv[0],
        '-p', 'face-crop',
        '-f', 'eigenfaces',
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '-g', 'grid',
        '--dry-run',
        '--user-directory', test_dir,
        '-b', 'dummy'
    ]

    print (facereclib.utils.command_line(parameters))

    # run the test; should not execute anything...
    from facereclib.script.faceverify_lfw import main
    main(parameters)
    shutil.rmtree(test_dir)


  def test05_faceverify_gbu_local(self):
    # try to import the gbu database
    try:
      facereclib.utils.resources.load_resource('gbu','database')
    except Exception as e:
      raise SkipTest("The resource for database 'gbu' could not be loaded; probably you didn't define the 'bob.db.gbu' in your *buildout.cfg*. Here is the import error: '%s'" % e)

    test_dir = tempfile.mkdtemp(prefix='frltest_')
    # define dummy parameters
    parameters = [
        sys.argv[0],
        '-p', 'face-crop',
        '-f', 'eigenfaces',
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '--dry-run',
        '--user-directory', test_dir,
        '-b', 'dummy'
    ]

    print (facereclib.utils.command_line(parameters))

    # run the test; should not execute anything...
    from facereclib.script.faceverify_gbu import main
    main(parameters)
    shutil.rmtree(test_dir)


  def test06_faceverify_gbu_grid(self):
    self.grid_available()
    # try to import the gbu database
    try:
      facereclib.utils.resources.load_resource('gbu','database')
    except Exception as e:
      raise SkipTest("The resource for database 'gbu' could not be loaded; probably you didn't define the 'bob.db.gbu' in your *buildout.cfg*. Here is the import error: '%s'" % e)

    test_dir = tempfile.mkdtemp(prefix='frltest_')
    # define dummy parameters
    parameters = [
        sys.argv[0],
        '-p', 'face-crop',
        '-f', 'eigenfaces',
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '-g', 'grid',
        '--dry-run',
        '--user-directory', test_dir,
        '-b', 'dummy'
    ]

    print (facereclib.utils.command_line(parameters))

    # run the test; should not execute anything...
    from facereclib.script.faceverify_gbu import main
    main(parameters)
    shutil.rmtree(test_dir)


  def test10_faceverify_file_set(self):
    test_dir = tempfile.mkdtemp(prefix='frltest_')
    # define dummy parameters
    parameters = [
        '-d', os.path.join(base_dir, 'scripts', 'fileset_Test.py'),
        '-p', os.path.join(config_dir, 'preprocessing', 'face_crop.py'),
        '-f', os.path.join(config_dir, 'features', 'eigenfaces.py'),
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '--zt-norm',
        '-b', 'test',
        '--temp-directory', test_dir,
        '--user-directory', test_dir
    ]

    print (facereclib.utils.command_line(parameters))

    self.__face_verify__(parameters, test_dir, 'test', ref_modifier="-fileset")


  def test11_baselines_api(self):
    self.grid_available()
    # test that all of the baselines would execute
    from facereclib.script.baselines import available_databases, all_algorithms, main

    for database in available_databases:
      parameters = [sys.argv[0], '-d', database, '--dry-run']
      main(parameters)
      parameters.append('-g')
      main(parameters)
      parameters.extend(['-e', 'HTER'])
      main(parameters)

    for algorithm in all_algorithms:
      parameters = [sys.argv[0], '-a', algorithm, '--dry-run']
      main(parameters)
      parameters.append('-g')
      main(parameters)
      parameters.extend(['-e', 'HTER'])
      main(parameters)


  def test15_evaluate(self):
    # tests our 'evaluate' script using the reference files
    test_dir = tempfile.mkdtemp(prefix='frltest_')
    reference_files = ('scores-nonorm-dev', 'scores-ztnorm-dev')
    plots = [os.path.join(test_dir, '%s.pdf')%f for f in ['roc', 'cmc', 'det']]
    parameters = [
      '--dev-files', reference_files[0], reference_files[1],
      '--eval-files', reference_files[0], reference_files[1],
      '--directory', os.path.join(base_dir, 'scripts'),
      '--legends', 'no norm', 'ZT norm',
      '--criterion', 'HTER',
      '--roc', plots[0],
      '--det', plots[1],
      '--cmc', plots[2],
    ]

    # execute the script
    from facereclib.script.evaluate import main
    main(parameters)
    for i in range(3):
      self.assertTrue(os.path.exists(plots[i]))
      os.remove(plots[i])
    os.rmdir(test_dir)


  def test16_collect_results(self):
    # simply test that the collect_results script works
    test_dir = tempfile.mkdtemp(prefix='frltest_')
    from facereclib.script.collect_results import main
    main(['--directory', test_dir, '--sort', '--sort-key', 'dir', '--criterion', 'FAR', '--self-test'])
    os.rmdir(test_dir)


  def test21_parameter_script(self):
    self.grid_available()
    test_dir = tempfile.mkdtemp(prefix='frltest_')
    # tests that the parameter_test.py script works properly

    # first test without grid option
    parameters = [
        sys.argv[0],
        '-c', os.path.join(base_dir, 'scripts', 'parameter_Test.py'),
        '-d', os.path.join(base_dir, 'scripts', 'atnt_Test.py'),
        '-f', 'lgbphs',
        '-b', 'test_p',
        '-s', '.',
        '-T', test_dir,
        '-R', test_dir,
        '--', '--dry-run',
    ]
    from facereclib.script.parameter_test import main
    main(parameters)

    # number of jobs should be 12
    self.assertEqual(facereclib.script.parameter_test.task_count, 12)
    # but no job in the grid
    self.assertEqual(facereclib.script.parameter_test.job_count, 0)

    # now, in the grid...
    parameters = [
        sys.argv[0],
        '-c', os.path.join(base_dir, 'scripts', 'parameter_Test.py'),
        '-d', os.path.join(base_dir, 'scripts', 'atnt_Test.py'),
        '-f', 'lgbphs',
        '-b', 'test_p',
        '-i', '.',
        '-s', '.',
        '-T', test_dir,
        '-R', test_dir,
        '-g', 'grid',
        '--', '--dry-run',
    ]
    main(parameters)

    # number of jobs should be 12
    self.assertEqual(facereclib.script.parameter_test.task_count, 12)
    # number of jobs in the grid: 36 (including best possible re-use of files; minus preprocessing)
    self.assertEqual(facereclib.script.parameter_test.job_count, 36)

    shutil.rmtree(test_dir)
