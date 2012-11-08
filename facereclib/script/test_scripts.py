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
import shutil
import tempfile

from nose.plugins.skip import SkipTest

base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
config_dir = os.path.join(base_dir, 'facereclib', 'configurations')

class TestScript (unittest.TestCase):

  def __face_verify__(self, parameters, test_dir, sub_dir):
    import faceverify
    faceverify.main(parameters)

    # assert that the score file exists
    score_files = (os.path.join(test_dir, sub_dir, 'scores', 'Default', 'nonorm', 'scores-dev'), os.path.join(test_dir, sub_dir, 'scores', 'Default', 'ztnorm', 'scores-dev'))
    self.assertTrue(os.path.exists(score_files[0]))
    self.assertTrue(os.path.exists(score_files[1]))

    # also assert that the scores are still the same -- though they have no real meaning
    reference_files = (os.path.join(base_dir, 'testdata', 'scripts', 'scores-nonorm-dev'), os.path.join(base_dir, 'testdata', 'scripts', 'scores-ztnorm-dev'))

    for i in (0,1):

      f1 = open(score_files[i], 'r')
      f2 = open(reference_files[i], 'r')

      self.assertTrue(f1.read() == f2.read())
      f1.close()
      f2.close()

    shutil.rmtree(test_dir)



  def test01_faceverify_local(self):
    test_dir = tempfile.mkdtemp(prefix='frl_')
    # define dummy parameters
    parameters = [
        '-d', os.path.join(base_dir, 'testdata', 'scripts', 'atnt_Test.py'),
        '-p', os.path.join(config_dir, 'preprocessing', 'face_crop.py'),
        '-f', os.path.join(config_dir, 'features', 'eigenfaces.py'),
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '--zt-norm',
        '-b', 'test',
        '--temp-directory', test_dir,
        '--user-directory', test_dir
    ]

    print ' '.join(parameters)

    self.__face_verify__(parameters, test_dir, 'test')


  def test01a_faceverify_resources(self):
    test_dir = tempfile.mkdtemp(prefix='frl_')
    # define dummy parameters
    parameters = [
        '-d', os.path.join(base_dir, 'testdata', 'scripts', 'atnt_Test.py'),
        '-p', 'face-crop',
        '-f', 'eigenfaces',
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '--zt-norm',
        '-b', 'test_a',
        '--temp-directory', test_dir,
        '--user-directory', test_dir
    ]

    print ' '.join(parameters)

    self.__face_verify__(parameters, test_dir, 'test_a')



  def test01b_faceverify_commandline(self):
    test_dir = tempfile.mkdtemp(prefix='frl_')
    # define dummy parameters
    parameters = [
        '-d', os.path.join(base_dir, 'testdata', 'scripts', 'atnt_Test.py'),
        '-p', 'face-crop',
        '-f', 'facereclib.features.Eigenface(subspace_dimension', '=', '100)',
        '-t', 'facereclib.tools.DummyTool()',
        '--zt-norm',
        '-b', 'test_b',
        '--temp-directory', test_dir,
        '--user-directory', test_dir
    ]

    print ' '.join(parameters)

    self.__face_verify__(parameters, test_dir, 'test_b')




  def test01x_faceverify_fl(self):
    test_dir = tempfile.mkdtemp(prefix='frl_')
    # define dummy parameters
    parameters = [
        '-d', os.path.join(base_dir, 'testdata', 'databases', 'atnt_fl', 'atnt_fl_database.py'),
        '-p', os.path.join(config_dir, 'preprocessing', 'face_crop.py'),
        '-f', os.path.join(config_dir, 'features', 'eigenfaces.py'),
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '--zt-norm',
        '-b', 'test_x',
        '--temp-directory', test_dir,
        '--user-directory', test_dir
    ]

    print ' '.join(parameters)

    import faceverify
    faceverify.main(parameters)

    # assert that the score file exists
    score_files = (os.path.join(test_dir, 'test_x', 'scores', 'Default', 'nonorm', 'scores-dev'), os.path.join(test_dir, 'test_x', 'scores', 'Default', 'ztnorm', 'scores-dev'))
    self.assertTrue(os.path.exists(score_files[0]))
    self.assertTrue(os.path.exists(score_files[1]))

    # assert that the scores are are identical
    reference_files = (os.path.join(base_dir, 'testdata', 'scripts', 'scores-nonorm-dev'), os.path.join(base_dir, 'testdata', 'scripts', 'scores-ztnorm-dev'))

    import bob
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
    # define dummy parameters including the dry-run
    parameters = [
        '-d', 'atnt',
        '-p', 'face-crop',
        '-f', 'eigenfaces',
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '-g', 'grid',
        '--dry-run',
        '-b', 'dummy'
    ]

    print ' '.join(parameters)

    # run the test; should not execute anything...
    import faceverify
    faceverify.main(parameters)


  def test03_faceverify_lfw_local(self):
    # define dummy parameters
    parameters = [
        '-p', 'face-crop',
        '-f', 'eigenfaces',
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '--dry-run',
        '-b', 'dummy'
    ]

    print ' '.join(parameters)

    # run the test; should not execute anything...
    import faceverify_lfw
    faceverify_lfw.main(parameters)


  def test04_faceverify_lfw_grid(self):
    # define dummy parameters
    parameters = [
        '-p', 'face-crop',
        '-f', 'eigenfaces',
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '-g', 'grid',
        '--dry-run',
        '-b', 'dummy'
    ]

    print ' '.join(parameters)

    # run the test; should not execute anything...
    import faceverify_lfw
    faceverify_lfw.main(parameters)


  def test05_faceverify_gbu_local(self):
    # define dummy parameters
    parameters = [
        '-p', 'face-crop',
        '-f', 'eigenfaces',
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '--dry-run',
        '-b', 'dummy'
    ]

    print ' '.join(parameters)

    # run the test; should not execute anything...
    import faceverify_gbu
    faceverify_gbu.main(parameters)


  def test06_faceverify_gbu_grid(self):
    # define dummy parameters
    # define dummy parameters
    parameters = [
        '-p', 'face-crop',
        '-f', 'eigenfaces',
        '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
        '-g', 'grid',
        '--dry-run',
        '-b', 'dummy'
    ]

    print ' '.join(parameters)

    # run the test; should not execute anything...
    import faceverify_gbu
    faceverify_gbu.main(parameters)



  def test11_baselines_api(self):
    # test that all of the baselines would execute
    from facereclib.script.baselines import all_databases, all_algorithms, main

    for database in all_databases:
      parameters = ['-d', database, '--dry-run']
      main(parameters)
      parameters.append('-g')
      main(parameters)
      parameters.append('-e')
      main(parameters)

    for algorithm in all_algorithms:
      parameters = ['-a', algorithm, '--dry-run']
      main(parameters)
      parameters.append('-g')
      main(parameters)
      parameters.append('-e')
      main(parameters)



  def test21_parameter_script(self):
    test_dir = tempfile.mkdtemp(prefix='frl_')
    # tests that the parameter_test.py script works properly
    import parameter_test

    # first test without grid option
    parameters = [
        '-c', os.path.join(base_dir, 'testdata', 'scripts', 'parameter_Test.py'),
        '-d', os.path.join(base_dir, 'testdata', 'scripts', 'atnt_Test.py'),
        '-b', 'test_p',
        '-s', test_dir,
        '--', '--dry-run'
    ]
    parameter_test.main(parameters)

    # number of jobs should be 12
    self.assertEqual(parameter_test.task_count, 12)
    # but no job in the grid
    self.assertEqual(parameter_test.job_count, 0)

    # now, in the grid...
    parameters = [
        '-c', os.path.join(base_dir, 'testdata', 'scripts', 'parameter_Test.py'),
        '-d', os.path.join(base_dir, 'testdata', 'scripts', 'atnt_Test.py'),
        '-b', 'test_p',
        '-s', test_dir,
        '-g', 'grid',
        '--', '--dry-run'
    ]
    parameter_test.main(parameters)

    # number of jobs should be 12
    self.assertEqual(parameter_test.task_count, 12)
    # number of jobs in the grid: 42 (including best possible re-use of files)
    self.assertEqual(parameter_test.job_count, 42)

    shutil.rmtree(test_dir)
