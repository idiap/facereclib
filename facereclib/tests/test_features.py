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

import bob.io.base
import bob.learn.linear

import unittest
import os
import numpy
import math
import tempfile
import facereclib
from nose.plugins.skip import SkipTest

import pkg_resources

regenerate_refs = False


class FeatureExtractionTest(unittest.TestCase):

  def input_dir(self, file):
    return pkg_resources.resource_filename('facereclib.tests', os.path.join('preprocessing', file))

  def reference_dir(self, file):
    ref = pkg_resources.resource_filename('facereclib.tests', os.path.join('features', file))
    facereclib.utils.ensure_dir(os.path.dirname(ref))
    return ref

  def config(self, resource):
    return facereclib.utils.tests.configuration_file(resource, 'feature_extractor', 'features')


  def execute(self, extractor, data, reference, epsilon = 1e-5):
    # execute the preprocessor
    feature = extractor(data)
    if regenerate_refs:
      facereclib.utils.save(feature, self.reference_dir(reference))

    ref = facereclib.utils.load(self.reference_dir(reference))
    self.assertEqual(ref.shape, feature.shape)
    self.assertTrue((numpy.abs(ref - feature) < epsilon).all())
    return feature


  def test01_linearize(self):
    # read input
    data = facereclib.utils.load(self.input_dir('cropped.hdf5'))
    extractor = self.config('linearize')
    # extract feature
    feature = self.execute(extractor, data, 'linearize.hdf5')
    self.assertTrue(len(feature.shape) == 1)


  def test02_dct(self):
    # read input
    data = facereclib.utils.load(self.input_dir('cropped.hdf5'))
    extractor = self.config('dct')
    # extract feature
    feature = self.execute(extractor, data, 'dct_blocks.hdf5')
    self.assertEqual(len(feature.shape), 2)

    # also test extractor with tuple input
    extractor = facereclib.features.DCTBlocks((12,12), (11,11), 45)
    # extract feature
    feature = self.execute(extractor, data, 'dct_blocks.hdf5')
    self.assertEqual(len(feature.shape), 2)


  def test03_graphs(self):
    data = bob.io.base.load(self.input_dir('cropped.hdf5'))
    extractor = self.config('grid-graph')
    # execute extractor
    feature = extractor(data)
    if regenerate_refs:
      extractor.save_feature(feature, self.reference_dir('graph_regular.hdf5'))
    ref = extractor.read_feature(self.reference_dir('graph_regular.hdf5'))
    self.assertEqual(len(ref), len(feature))
    for i in range(len(ref)):
      self.assertTrue((numpy.abs(ref[i].jet - feature[i].jet) < 1e-5).all())

    # generate aligned graph extractor
    extractor = self.config('grid_graph_aligned')
    # execute extractor
    feature = extractor(data)
    if regenerate_refs:
      extractor.save_feature(feature, self.reference_dir('graph_aligned.hdf5'))
    ref = extractor.read_feature(self.reference_dir('graph_aligned.hdf5'))
    self.assertEqual(len(ref), len(feature))
    for i in range(len(ref)):
      self.assertTrue((numpy.abs(ref[i].jet - feature[i].jet) < 1e-5).all())

    # test the automatic computation of start node
    extractor = facereclib.features.GridGraph(
      gabor_sigma = math.sqrt(2.) * math.pi,
      node_distance = (10, 10),
      image_resolution = (80, 64)
    )
    self.assertEqual(len(extractor.m_graph.nodes), 48)
    self.assertTrue(extractor.m_graph.nodes[0] == (5, 7))
    self.assertTrue(extractor.m_graph.nodes[-1] == (75, 57))


  def test04_lgbphs(self):
    data = bob.io.base.load(self.input_dir('cropped.hdf5'))
    # just test if the config file loads correctly...
    extractor = self.config('lgbphs')
    self.assertTrue(isinstance(extractor, facereclib.features.LGBPHS))

    # in this test, we use a smaller setup of the LGBPHS features
    extractor = facereclib.features.LGBPHS(
        block_size = 8,
        block_overlap = 0,
        gabor_directions = 4,
        gabor_scales = 2,
        gabor_sigma = math.sqrt(2.) * math.pi,
        sparse_histogram = True
    )
    # execute feature extractor
    feature = self.execute(extractor, data, 'lgbphs_sparse.hdf5')
    self.assertEqual(len(feature.shape), 2) # we use sparse histogram by default

    # generate new non-sparse extractor
    extractor = facereclib.features.LGBPHS(
        block_size = 8,
        block_overlap = 0,
        gabor_directions = 4,
        gabor_scales = 2,
        gabor_sigma = math.sqrt(2.) * math.pi,
    )
    no_phase = self.execute(extractor, data, 'lgbphs_no_phase.hdf5')
    self.assertEqual(len(no_phase.shape), 1)

    # generate new graph without phases
    extractor = facereclib.features.LGBPHS(
        block_size = 8,
        block_overlap = 0,
        gabor_directions = 4,
        gabor_scales = 2,
        gabor_sigma = math.sqrt(2.) * math.pi,
        use_gabor_phases = True
    )
    with_phase = self.execute(extractor, data, 'lgbphs_with_phase.hdf5')
    self.assertTrue(len(with_phase.shape) == 1)
    self.assertEqual(no_phase.shape[0]*2, with_phase.shape[0])


  def test05_sift_key_points(self):
    # check if VLSIFT is available
    import bob.ip.base
    if not hasattr(bob.ip.base, "VLSIFT"):
      raise SkipTest("VLSIFT is not part of bob.ip.base; maybe SIFT headers aren't installed in your system?")

    # we need the preprocessor tool to actually read the data
    preprocessor = facereclib.preprocessing.Keypoints()
    data = preprocessor.read_data(self.input_dir('key_points.hdf5'))
    # now, we extract features from it
    extractor = self.config('sift')
    feature = self.execute(extractor, data, 'sift.hdf5', epsilon=1e-4)
    self.assertEqual(len(feature.shape), 1)


  def test06_eigenface(self):
    # just test if the config file loads correctly...
    extractor = self.config('eigenfaces')
    self.assertTrue(isinstance(extractor, facereclib.features.Eigenface))

    # create extractor with a smaller number of kept eigenfaces
    extractor = facereclib.features.Eigenface(subspace_dimension = 5)
    self.assertTrue(extractor.requires_training)

    # we read the test data (so that we have a length)
    data = bob.io.base.load(self.input_dir('cropped.hdf5'))
    # we have to train the eigenface extractor, so we generate some data
    train_data = facereclib.utils.tests.random_training_set(data.shape, 400, 0., 255.)
    t = tempfile.mkstemp('pca.hdf5', prefix='frltest_')[1]
    extractor.train(train_data, t)
    if regenerate_refs:
      import shutil
      shutil.copy2(t, self.reference_dir('eigenface_extractor.hdf5'))

    extractor.load(self.reference_dir('eigenface_extractor.hdf5'))
    # compare the resulting machines
    new_machine = bob.learn.linear.Machine(bob.io.base.HDF5File(t))
    self.assertEqual(extractor.m_machine.shape, new_machine.shape)
    # ... rotation direction might change, hence either the sum or the difference should be 0
    for i in range(5):
      self.assertTrue(numpy.abs(extractor.m_machine.weights[:,i] - new_machine.weights[:,i] < 1e-5).all() or numpy.abs(extractor.m_machine.weights[:,i] + new_machine.weights[:,i] < 1e-5).all())
    os.remove(t)

    # now, we can execute the extractor and check that the feature is still identical
    feature = self.execute(extractor, data, 'eigenface.hdf5')
    self.assertEqual(len(feature.shape), 1)
