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
import numpy
import tempfile
import facereclib
import bob
from nose.plugins.skip import SkipTest

regenerate_refs = False

class FeatureExtractionTest(unittest.TestCase):

  def input_dir(self, file):
    return os.path.join('testdata', 'preprocessing', file)

  def reference_dir(self, file):
    dir = os.path.join('testdata', 'features')
    facereclib.utils.ensure_dir(dir)
    return os.path.join(dir, file)

  def config(self, file):
    return imp.load_source('config', os.path.join('config', 'features', file))


  def train_set(self, feature, count = 50, a = 0, b = 1):
    # generate a random sequence of features
    numpy.random.seed(42)
    return [numpy.random.random(feature.shape) * (b - a) + a for i in range(count)]


  def execute(self, extractor, image, reference):
    # execute the preprocessor
    feature = extractor(image)
    if regenerate_refs:
      bob.io.save(feature, self.reference_dir(reference))

    self.assertTrue((numpy.abs(bob.io.load(self.reference_dir(reference)) - feature) < 1e-5).all())
    return feature



  def test01_linearize(self):
    # read input
    image = bob.io.load(self.input_dir('cropped.hdf5'))
    config = self.config('linearize.py')

    # generate extractor
    extractor = config.feature_extractor(config)

    # extract feature
    feature = self.execute(extractor, image, 'linearize.hdf5')
    self.assertTrue(len(feature.shape) == 1)


  def test02_dct(self):
    # read input
    image = bob.io.load(self.input_dir('cropped.hdf5'))
    config = self.config('dct_blocks.py')

    # generate extractor
    extractor = config.feature_extractor(config)

    # extract feature
    feature = self.execute(extractor, image, 'dct_blocks.hdf5')
    self.assertEqual(len(feature.shape), 2)


  def test03_graphs(self):
    image = bob.io.load(self.input_dir('cropped.hdf5'))
    config = self.config('grid_graph.py')

    # generate extractor
    extractor = config.feature_extractor(config)

    # execute
    feature = self.execute(extractor, image, 'graph_with_phase.hdf5')
    self.assertEqual(len(feature.shape), 3)

    # generate new graph without phases
    config.EXTRACT_GABOR_PHASES = False
    extractor = config.feature_extractor(config)
    feature = self.execute(extractor, image, 'graph_no_phase.hdf5')
    self.assertEqual(len(feature.shape), 2)


  def test04_lgbphs(self):
    image = bob.io.load(self.input_dir('cropped.hdf5'))
    config = self.config('lgbphs.py')
    # generate smaller features for test purposes
    config.GABOR_DIRECTIONS = 4
    config.GABOR_SCALES = 2
    config.BLOCK_Y_OVERLAP = 0
    config.BLOCK_X_OVERLAP = 0

    # generate extractor
    extractor = config.feature_extractor(config)

    # execute
    feature = self.execute(extractor, image, 'lgbphs_sparse.hdf5')
    self.assertEqual(len(feature.shape), 2) # we use sparse histogram by default

    # generate new non-sparse
    config.USE_SPARSE_HISTOGRAM = False
    extractor = config.feature_extractor(config)
    no_phase = self.execute(extractor, image, 'lgbphs_no_phase.hdf5')
    self.assertEqual(len(no_phase.shape), 1)

    # generate new graph without phases
    config.USE_GABOR_PHASES = True
    extractor = config.feature_extractor(config)
    with_phase = self.execute(extractor, image, 'lgbphs_with_phase.hdf5')
    self.assertTrue(len(with_phase.shape) == 1)
    self.assertEqual(no_phase.shape[0]*2, with_phase.shape[0])


  def test05_dct_video(self):
    raise SkipTest("Video tests are currently skipped.")
    # we need the preprocessor tool to actually read the data
    config = self.pre_config('tan_triggs_video.py')
    preprocessor = config.preprocessor(config)
    video = preprocessor.read_image(self.input_dir('video.hdf5'))

    # now, we extract features from it
    config = self.config('dct_blocks_video.py')
    extractor = config.feature_extractor(config)
    feature = extractor(video)

    if regenerate_refs:
      feature.save(bob.io.HDF5File(self.reference_dir('dct_video.hdf5'), 'w'))

    reference = extractor.read_feature(self.reference_dir('dct_video.hdf5'))

    self.assertEqual(feature, reference)


  def test06_sift_key_points(self):
    # we need the preprocessor tool to actually read the data
    preprocessor = facereclib.preprocessing.Keypoints()
    image = preprocessor.read_image(self.input_dir('key_points.hdf5'))

    # now, we extract features from it
    config = self.config('sift_keypoints.py')
    extractor = config.feature_extractor(config)

    feature = self.execute(extractor, image, 'sift.hdf5')
    self.assertEqual(len(feature.shape), 1)


  def test07_eigenface(self):
    # first, read the config file
    config = self.config('eigenfaces.py')
    config.SUBSPACE_DIMENSION = 5
    extractor = config.feature_extractor(config)

    self.assertTrue(extractor.requires_training)

    # we read the test image (so that we have a length)
    image = bob.io.load(self.input_dir('cropped.hdf5'))

    # we have to train the eigenface extractor, so we generate some data
    train_data = self.train_set(image, 400, 0., 255.)

    t = tempfile.mkstemp('pca.hdf5')[1]
    extractor.train(train_data, t)

    if regenerate_refs:
      import shutil
      shutil.copy2(t, self.reference_dir('eigenface_extractor.hdf5'))

    extractor.load(self.reference_dir('eigenface_extractor.hdf5'))

    # compare the resulting machines
    new_machine = bob.machine.LinearMachine(bob.io.HDF5File(t))

    self.assertEqual(extractor.m_machine.shape, new_machine.shape)
    self.assertTrue(numpy.abs(extractor.m_machine.weights - new_machine.weights < 1e-5).all())
    os.remove(t)

    # now, we can execute the extractor and check that the feature is still identical
    feature = self.execute(extractor, image, 'eigenface.hdf5')
    self.assertEqual(len(feature.shape), 1)


