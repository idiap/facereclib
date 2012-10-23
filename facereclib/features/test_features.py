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
import numpy
import math
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
    return facereclib.utils.read_config_file(os.path.join('config', 'features', file), 'feature_extractor')


  def train_set(self, feature, count = 50, a = 0, b = 1):
    # generate a random sequence of features
    numpy.random.seed(42)
    return [numpy.random.random(feature.shape) * (b - a) + a for i in range(count)]


  def execute(self, extractor, image, reference):
    # execute the preprocessor
    feature = extractor(image)
    if regenerate_refs:
      bob.io.save(feature, self.reference_dir(reference))

    ref = bob.io.load(self.reference_dir(reference))
    self.assertEqual(ref.shape, feature.shape)
    self.assertTrue((numpy.abs(ref - feature) < 1e-5).all())
    return feature


  def test01_linearize(self):
    # read input
    image = bob.io.load(self.input_dir('cropped.hdf5'))
    extractor = self.config('linearize.py')
    # extract feature
    feature = self.execute(extractor, image, 'linearize.hdf5')
    self.assertTrue(len(feature.shape) == 1)


  def test02_dct(self):
    # read input
    image = bob.io.load(self.input_dir('cropped.hdf5'))
    extractor = self.config('dct_blocks.py')
    # extract feature
    feature = self.execute(extractor, image, 'dct_blocks.hdf5')
    self.assertEqual(len(feature.shape), 2)

    # also test extractor with tuple input
    extractor = facereclib.features.DCTBlocks((12,12), (11,11), 45)
    # extract feature
    feature = self.execute(extractor, image, 'dct_blocks.hdf5')
    self.assertEqual(len(feature.shape), 2)


  def test02a_dct_video(self):
    # test that at least the config file can be read
    extractor = self.config('dct_blocks_video.py')
    self.assertTrue(isinstance(extractor, facereclib.features.DCTBlocksVideo))
    raise SkipTest("Video tests are currently skipped.")
    # we need the preprocessor tool to actually read the data
    preprocessor = facereclib.utils.read_config_file(os.path.join('config', 'preprocessor', 'tan_triggs_video.py'), 'preprocessor')
    video = preprocessor.read_image(self.input_dir('video.hdf5'))

    # now, we extract features from it
    feature = extractor(video)
    if regenerate_refs:
      feature.save(bob.io.HDF5File(self.reference_dir('dct_video.hdf5'), 'w'))
    reference = extractor.read_feature(self.reference_dir('dct_video.hdf5'))
    self.assertEqual(feature, reference)


  def test03_graphs(self):
    image = bob.io.load(self.input_dir('cropped.hdf5'))
    extractor = self.config('grid_graph.py')
    # execute extractor
    feature = self.execute(extractor, image, 'graph_with_phase.hdf5')
    self.assertEqual(len(feature.shape), 3)

    # generate new graph extractor without phases
    extractor = facereclib.features.GridGraph(
      gabor_sigma = math.sqrt(2.) * math.pi,
      extract_gabor_phases = False,
      first_node = (6, 6),
      last_node = (image.shape[0] - 6, image.shape[1] - 6),
      node_distance = (4, 4)
    )
    feature = self.execute(extractor, image, 'graph_no_phase.hdf5')
    self.assertEqual(len(feature.shape), 2)

    # generate aligned graph extractor
    extractor = self.config('grid_graph_aligned.py')
    feature = self.execute(extractor, image, 'graph_aligned.hdf5')
    self.assertEqual(len(feature.shape), 3)


  def test04_lgbphs(self):
    image = bob.io.load(self.input_dir('cropped.hdf5'))
    # just test if the config file loads correctly...
    extractor = self.config('lgbphs.py')
    self.assertTrue(isinstance(extractor, facereclib.features.LGBPHS))

    # in this test, we use a smaller setup of the LGBPHS features
    extractor = facereclib.features.LGBPHS(
        block_size = 10,
        block_overlap = 0,
        gabor_directions = 4,
        gabor_scales = 2,
        gabor_sigma = math.sqrt(2.) * math.pi,
        sparse_histogram = True
    )
    # execute feature extractor
    feature = self.execute(extractor, image, 'lgbphs_sparse.hdf5')
    self.assertEqual(len(feature.shape), 2) # we use sparse histogram by default

    # generate new non-sparse extractor
    extractor = facereclib.features.LGBPHS(
        block_size = 10,
        block_overlap = 0,
        gabor_directions = 4,
        gabor_scales = 2,
        gabor_sigma = math.sqrt(2.) * math.pi,
    )
    no_phase = self.execute(extractor, image, 'lgbphs_no_phase.hdf5')
    self.assertEqual(len(no_phase.shape), 1)

    # generate new graph without phases
    extractor = facereclib.features.LGBPHS(
        block_size = 10,
        block_overlap = 0,
        gabor_directions = 4,
        gabor_scales = 2,
        gabor_sigma = math.sqrt(2.) * math.pi,
        use_gabor_phases = True
    )
    with_phase = self.execute(extractor, image, 'lgbphs_with_phase.hdf5')
    self.assertTrue(len(with_phase.shape) == 1)
    self.assertEqual(no_phase.shape[0]*2, with_phase.shape[0])


  def test05_sift_key_points(self):
    # we need the preprocessor tool to actually read the data
    preprocessor = facereclib.preprocessing.Keypoints()
    image = preprocessor.read_image(self.input_dir('key_points.hdf5'))
    # now, we extract features from it
    extractor = self.config('sift_keypoints.py')
    feature = self.execute(extractor, image, 'sift.hdf5')
    self.assertEqual(len(feature.shape), 1)


  def test06_eigenface(self):
    # just test if the config file loads correctly...
    extractor = self.config('eigenfaces.py')
    self.assertTrue(isinstance(extractor, facereclib.features.Eigenface))

    # create extractor with a smaller number of kept eigenfaces
    extractor = facereclib.features.Eigenface(subspace_dimension = 5)
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
