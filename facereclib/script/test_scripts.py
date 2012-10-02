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
config_dir = os.path.join(base_dir, 'config')

class TestScript (unittest.TestCase):


  def test01_faceverify_local(self):
    test_dir = tempfile.mkdtemp(prefix='frl_')
    # define dummy parameters
    parameters = ['-d', os.path.join(config_dir, 'database', 'atnt_Default.py'),
                  '-p', os.path.join(config_dir, 'preprocessing', 'face_crop.py'),
                  '-f', os.path.join(config_dir, 'features', 'eigenfaces.py'),
                  '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
                  '-b', 'dummy',
                  '--temp-directory', test_dir,
                  '--user-directory', test_dir
                  ]

    print ' '.join(parameters)

    import faceverify
    verif_args = faceverify.parse_args(parameters)
    job_ids = faceverify.face_verify(verif_args)

    # assert that the score file exists
    score_file = os.path.join(test_dir, 'dummy', 'scores', 'Default', 'nonorm', 'scores-dev')
    self.assertTrue(os.path.exists(score_file))

    # also assert that the scores are still the same -- though they have no real meaning
    reference_file = os.path.join(base_dir, 'testdata', 'scripts', 'scores-nonorm-dev')

    f1 = open(score_file, 'r')
    f2 = open(reference_file, 'r')

    self.assertTrue(f1.read() == f2.read())
    f1.close()
    f2.close()

    shutil.rmtree(test_dir)

  def test02_faceverify_grid(self):
    # define dummy parameters including the dry-run
    parameters = ['-d', os.path.join(config_dir, 'database', 'atnt_Default.py'),
                  '-p', os.path.join(config_dir, 'preprocessing', 'face_crop.py'),
                  '-f', os.path.join(config_dir, 'features', 'eigenfaces.py'),
                  '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
                  '-g', os.path.join(config_dir, 'grid', 'grid.py'),
                  '--dry-run',
                  '-b', 'dummy'
                  ]

    print ' '.join(parameters)

    # run the test; should not execute anything...
    import faceverify
    verif_args = faceverify.parse_args(parameters)
    job_ids = faceverify.face_verify(verif_args)


  def test05_faceverify_gbu_local(self):
    # define dummy parameters
    parameters = ['-d', os.path.join(config_dir, 'database', 'gbu_Good.py'),
                  '-p', os.path.join(config_dir, 'preprocessing', 'face_crop.py'),
                  '-f', os.path.join(config_dir, 'features', 'eigenfaces.py'),
                  '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
                  '--dry-run',
                  '-b', 'dummy'
                  ]

    print ' '.join(parameters)

    # run the test; should not execute anything...
    import faceverify_gbu
    verif_args = faceverify_gbu.parse_args(parameters)
    job_ids = faceverify_gbu.face_verify(verif_args)


  def test06_faceverify_gbu_grid(self):
    # define dummy parameters
    parameters = ['-d', os.path.join(config_dir, 'database', 'gbu_Good.py'),
                  '-p', os.path.join(config_dir, 'preprocessing', 'face_crop.py'),
                  '-f', os.path.join(config_dir, 'features', 'eigenfaces.py'),
                  '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
                  '-g', os.path.join(config_dir, 'grid', 'grid.py'),
                  '--dry-run',
                  '-b', 'dummy'
                  ]

    print ' '.join(parameters)

    # run the test; should not execute anything...
    import faceverify_gbu
    verif_args = faceverify_gbu.parse_args(parameters)
    job_ids = faceverify_gbu.face_verify(verif_args)


  def test03_faceverify_lfw_local(self):
    # define dummy parameters
    parameters = ['-d', os.path.join(config_dir, 'database', 'lfw_view1.py'),
                  '-p', os.path.join(config_dir, 'preprocessing', 'face_crop.py'),
                  '-f', os.path.join(config_dir, 'features', 'eigenfaces.py'),
                  '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
                  '--dry-run',
                  '-b', 'dummy'
                  ]

    print ' '.join(parameters)

    # run the test; should not execute anything...
    import faceverify_lfw
    verif_args = faceverify_lfw.parse_args(parameters)
    job_ids = faceverify_lfw.face_verify(verif_args)


  def test04_faceverify_lfw_grid(self):
    # define dummy parameters
    parameters = ['-d', os.path.join(config_dir, 'database', 'lfw_view1.py'),
                  '-p', os.path.join(config_dir, 'preprocessing', 'face_crop.py'),
                  '-f', os.path.join(config_dir, 'features', 'eigenfaces.py'),
                  '-t', os.path.join(config_dir, 'tools', 'dummy.py'),
                  '-g', os.path.join(config_dir, 'grid', 'grid.py'),
                  '--dry-run',
                  '-b', 'dummy'
                  ]

    print ' '.join(parameters)

    # run the test; should not execute anything...
    import faceverify_lfw
    verif_args = faceverify_lfw.parse_args(parameters)
    job_ids = faceverify_lfw.face_verify(verif_args)

