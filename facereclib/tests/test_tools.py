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
import tempfile
import facereclib
import bob
from nose.plugins.skip import SkipTest

import pkg_resources

regenerate_refs = False
seed_value = 5489

class ToolTest(unittest.TestCase):

  def input_dir(self, file):
    return pkg_resources.resource_filename('facereclib.tests', os.path.join('features', file))

  def reference_dir(self, file):
    ref = pkg_resources.resource_filename('facereclib.tests', os.path.join('tools', file))
    facereclib.utils.ensure_dir(os.path.dirname(ref))
    return ref

  def config(self, resource):
    return facereclib.utils.tests.configuration_file(resource, 'tool', 'tools')


  def compare(self, feature, reference):
    # execute the preprocessor
    if regenerate_refs:
      bob.io.save(feature, self.reference_dir(reference))

    self.assertTrue((numpy.abs(bob.io.load(self.reference_dir(reference)) - feature) < 1e-5).all())


  def train_gmm_stats(self, feature_file, count = 50, minimum = 0, maximum = 1):
    # generate a random sequence of GMM-Stats features
    numpy.random.seed(42)
    train_set = []
    f = bob.io.HDF5File(feature_file)
    for i in range(count):
      per_id = []
      for j in range(count):
        gmm_stats = bob.machine.GMMStats(f)
        gmm_stats.sum_px = numpy.random.random(gmm_stats.sum_px.shape) * (maximum - minimum) + minimum
        gmm_stats.sum_pxx = numpy.random.random(gmm_stats.sum_pxx.shape) * (maximum - minimum) + minimum
        per_id.append(gmm_stats)
      train_set.append(per_id)
    return train_set



  def test01_gabor_jet(self):
    # read input
    feature = bob.io.load(self.input_dir('graph_with_phase.hdf5'))
    tool = self.config('gabor-jet')
    self.assertFalse(tool.performs_projection)
    self.assertFalse(tool.requires_enroller_training)

    # enroll
    model = tool.enroll([feature])
    self.compare(model, 'graph_model.hdf5')

    # score
    sim = tool.score(model, feature)
    self.assertAlmostEqual(sim, 1.)
    self.assertAlmostEqual(tool.score_for_multiple_probes(model, [feature, feature]), 1.)


  def test02_lgbphs(self):
    # read input
    feature1 = bob.io.load(self.input_dir('lgbphs_sparse.hdf5'))
    feature2 = bob.io.load(self.input_dir('lgbphs_no_phase.hdf5'))
    tool = self.config('lgbphs')
    self.assertFalse(tool.performs_projection)
    self.assertFalse(tool.requires_enroller_training)

    # enroll model
    model = tool.enroll([feature1])
    self.compare(model, 'lgbphs_model.hdf5')

    # score
    sim = tool.score(model, feature2)
    self.assertAlmostEqual(sim, 33600.0)
    self.assertAlmostEqual(tool.score_for_multiple_probes(model, [feature2, feature2]), sim)


  def test03_pca(self):
    # read input
    feature = bob.io.load(self.input_dir('linearize.hdf5'))
    # assure that the config file is read
    tool = self.config('pca')
    self.assertTrue(isinstance(tool, facereclib.tools.PCA))

    # generate tool with a lower number of dimensions
    tool = facereclib.tools.PCA(10)
    self.assertTrue(tool.performs_projection)
    self.assertTrue(tool.requires_projector_training)
    self.assertTrue(tool.use_projected_features_for_enrollment)
    self.assertFalse(tool.split_training_features_by_client)

    # train the projector
    t = tempfile.mkstemp('pca.hdf5', prefix='frltest_')[1]
    tool.train_projector(facereclib.utils.tests.random_training_set(feature.shape, count=400, minimum=0., maximum=255.), t)
    if regenerate_refs:
      import shutil
      shutil.copy2(t, self.reference_dir('pca_projector.hdf5'))

    # load the projector file
    tool.load_projector(self.reference_dir('pca_projector.hdf5'))
    # compare the resulting machines
    f = bob.io.HDF5File(t)
    new_variances = f.read("Eigenvalues")
    f.cd("/Machine")
    new_machine = bob.machine.LinearMachine(f)
    del f
    self.assertEqual(tool.m_variances.shape, new_variances.shape)
    self.assertTrue(numpy.abs(tool.m_variances - new_variances < 1e-5).all())
    self.assertEqual(tool.m_machine.shape, new_machine.shape)
    # ... rotation direction might change, hence either the sum or the difference should be 0
    for i in range(10):
      self.assertTrue(numpy.abs(tool.m_machine.weights[:,i] - new_machine.weights[:,i] < 1e-5).all() or numpy.abs(tool.m_machine.weights[:,i] + new_machine.weights[:,i] < 1e-5).all())
    os.remove(t)

    # project feature
    projected = tool.project(feature)
    self.compare(projected, 'pca_feature.hdf5')
    self.assertTrue(len(projected.shape) == 1)

    # enroll model
    model = tool.enroll([projected])
    self.compare(model, 'pca_model.hdf5')
    sim = tool.score(model, projected)
    self.assertAlmostEqual(sim, 0.)

    # test the calculation of the subspace dimension based on percentage of variance
    tool = facereclib.tools.PCA(.9)
    tool.train_projector(facereclib.utils.tests.random_training_set(feature.shape, count=400, minimum=0., maximum=255.), t)
    self.assertEqual(tool.m_subspace_dim, 334)
    tool.load_projector(t)
    os.remove(t)
    projected = tool.project(feature)
    model = tool.enroll([projected, projected])
    self.assertTrue(model.shape == (2,334))
    self.assertAlmostEqual(tool.score(model, projected), 0.)
    self.assertAlmostEqual(tool.score_for_multiple_probes(model, [projected, projected]), 0.)


  def test04_lda(self):
    # read input
    feature = bob.io.load(self.input_dir('linearize.hdf5'))
    # assure that the config file is loadable
    tool = self.config('lda')
    self.assertTrue(isinstance(tool, facereclib.tools.LDA))
    # assure that the config file is loadable
    tool = self.config('pca+lda')
    self.assertTrue(isinstance(tool, facereclib.tools.LDA))

    # here we use a reduced tool, using the scaled Euclidean distance (mahalanobis) from scipy
    import scipy.spatial
    tool = facereclib.tools.LDA(5, 10, scipy.spatial.distance.seuclidean, True, True)
    self.assertTrue(tool.performs_projection)
    self.assertTrue(tool.requires_projector_training)
    self.assertTrue(tool.use_projected_features_for_enrollment)
    self.assertTrue(tool.split_training_features_by_client)

    # train the projector
    t = tempfile.mkstemp('pca+lda.hdf5', prefix='frltest_')[1]
    tool.train_projector(facereclib.utils.tests.random_training_set_by_id(feature.shape, count=20, minimum=0., maximum=255.), t)
    if regenerate_refs:
      import shutil
      shutil.copy2(t, self.reference_dir('pca+lda_projector.hdf5'))

    # load the projector file
    tool.load_projector(self.reference_dir('pca+lda_projector.hdf5'))
    # compare the resulting machines
    f = bob.io.HDF5File(t)
    new_variances = f.read("Eigenvalues")
    f.cd("/Machine")
    new_machine = bob.machine.LinearMachine(f)
    del f
    self.assertEqual(tool.m_machine.shape, new_machine.shape)
    self.assertTrue(numpy.abs(tool.m_variances - new_variances < 1e-5).all())
    # ... rotation direction might change, hence either the sum or the difference should be 0
    for i in range(5):
      self.assertTrue(numpy.abs(tool.m_machine.weights[:,i] - new_machine.weights[:,i] < 1e-5).all() or numpy.abs(tool.m_machine.weights[:,i] + new_machine.weights[:,i] < 1e-5).all())
    os.remove(t)

    # project feature
    projected = tool.project(feature)
    self.compare(projected, 'pca+lda_feature.hdf5')
    self.assertTrue(len(projected.shape) == 1)

    # enroll model
    model = tool.enroll([projected])
    self.compare(model, 'pca+lda_model.hdf5')
    self.assertTrue(model.shape == (1,5))

    # score
    sim = tool.score(model, projected)
    self.assertAlmostEqual(sim, 0.)

    # test the calculation of the subspace dimension based on percentage of variance,
    # and the usage of a different way to compute the final score in case of multiple features per model
    tool = facereclib.tools.LDA(5, .9, multiple_model_scoring = 'median')
    tool.train_projector(facereclib.utils.tests.random_training_set_by_id(feature.shape, count=20, minimum=0., maximum=255.), t)
    self.assertEqual(tool.m_pca_subspace, 334)
    tool.load_projector(t)
    os.remove(t)
    projected = tool.project(feature)
    model = tool.enroll([projected, projected])
    self.assertTrue(model.shape == (2,5))
    self.assertAlmostEqual(tool.score(model, projected), 0.)
    self.assertAlmostEqual(tool.score_for_multiple_probes(model, [projected, projected]), 0.)


  def test05_bic(self):
    # read input
    feature = bob.io.load(self.input_dir('linearize.hdf5'))
    # check that the config file is readable
    tool = self.config('bic')
    self.assertTrue(isinstance(tool, facereclib.tools.BIC))

    # here, we use a reduced complexity for test purposes
    tool = facereclib.tools.BIC(numpy.subtract, 100, (5,7))
    self.assertFalse(tool.performs_projection)
    self.assertTrue(tool.requires_enroller_training)

    # train the enroller
    t = tempfile.mkstemp('bic.hdf5', prefix='frltest_')[1]
    tool.train_enroller(facereclib.utils.tests.random_training_set_by_id(feature.shape, count=10, minimum=0., maximum=255.), t)
    if regenerate_refs:
      import shutil
      shutil.copy2(t, self.reference_dir('bic_enroller.hdf5'))

    # load the projector file
    tool.load_enroller(self.reference_dir('bic_enroller.hdf5'))
    # compare the resulting machines
    new_machine = bob.machine.BICMachine(tool.m_use_dffs)
    new_machine.load(bob.io.HDF5File(t))
    self.assertTrue(tool.m_bic_machine.is_similar_to(new_machine))
    os.remove(t)

    # enroll model
    model = tool.enroll([feature])
    self.compare(model, 'bic_model.hdf5')

    # score and compare to the weird reference score ...
    sim = tool.score(model, feature)
    self.assertAlmostEqual(sim, 0.31276072)

    # now, test without PCA
    tool = facereclib.tools.BIC(numpy.subtract, 100)
    # train the enroller
    t = tempfile.mkstemp('iec.hdf5', prefix='frltest_')[1]
    tool.train_enroller(facereclib.utils.tests.random_training_set_by_id(feature.shape, count=10, minimum=0., maximum=255.), t)
    if regenerate_refs:
      import shutil
      shutil.copy2(t, self.reference_dir('iec_enroller.hdf5'))

    # load the projector file
    tool.load_enroller(self.reference_dir('iec_enroller.hdf5'))
    # compare the resulting machines
    new_machine = bob.machine.BICMachine(tool.m_use_dffs)
    new_machine.load(bob.io.HDF5File(t))
    self.assertTrue(tool.m_bic_machine.is_similar_to(new_machine))
    os.remove(t)

    # score and compare to the weird reference score ...
    sim = tool.score(model, feature)
    self.assertAlmostEqual(sim, 0.4070329180)


  def test06_gmm(self):
    # read input
    feature = bob.io.load(self.input_dir('dct_blocks.hdf5'))
    # assure that the config file is readable
    tool = self.config('gmm')
    self.assertTrue(isinstance(tool, facereclib.tools.UBMGMM))

    # here, we use a reduced complexity for test purposes
    tool = facereclib.tools.UBMGMM(
        number_of_gaussians = 2,
        k_means_training_iterations = 1,
        gmm_training_iterations = 1,
        INIT_SEED = seed_value,
    )
    self.assertTrue(tool.performs_projection)
    self.assertTrue(tool.requires_projector_training)
    self.assertFalse(tool.use_projected_features_for_enrollment)
    self.assertFalse(tool.split_training_features_by_client)

    # train the projector
    t = tempfile.mkstemp('ubm.hdf5', prefix='frltest_')[1]
    tool.train_projector(facereclib.utils.tests.random_training_set(feature.shape, count=5, minimum=-5., maximum=5.), t)
    if regenerate_refs:
      import shutil
      shutil.copy2(t, self.reference_dir('gmm_projector.hdf5'))

    # load the projector file
    tool.load_projector(self.reference_dir('gmm_projector.hdf5'))
    # compare GMM projector with reference
    new_machine = bob.machine.GMMMachine(bob.io.HDF5File(t))
    self.assertTrue(tool.m_ubm.is_similar_to(new_machine))
    os.remove(t)

    # project the feature
    projected = tool.project(feature)
    if regenerate_refs:
      projected.save(bob.io.HDF5File(self.reference_dir('gmm_feature.hdf5'), 'w'))
    probe = tool.read_probe(self.reference_dir('gmm_feature.hdf5'))
    self.assertTrue(projected.is_similar_to(probe))

    # enroll model with the unprojected feature
    model = tool.enroll([feature])
    if regenerate_refs:
      model.save(bob.io.HDF5File(self.reference_dir('gmm_model.hdf5'), 'w'))
    reference_model = tool.read_model(self.reference_dir('gmm_model.hdf5'))
    self.assertTrue(model.is_similar_to(reference_model))

    # score with projected feature and compare to the weird reference score ...
    sim = tool.score(reference_model, probe)
    self.assertAlmostEqual(sim, 0.25472347774)
    self.assertAlmostEqual(tool.score_for_multiple_probes(model, [probe, probe]), sim)


  def test06a_gmm_regular(self):
    # read input
    feature = bob.io.load(self.input_dir('dct_blocks.hdf5'))
    # assure that the config file is readable
    tool = self.config('ubm_gmm_regular_scoring')
    self.assertTrue(isinstance(tool, facereclib.tools.UBMGMMRegular))

    # here, we use a reduced complexity for test purposes
    tool = facereclib.tools.UBMGMMRegular(
        number_of_gaussians = 2,
        k_means_training_iterations = 1,
        gmm_training_iterations = 1,
        INIT_SEED = seed_value
    )
    self.assertFalse(tool.performs_projection)
    self.assertTrue(tool.requires_enroller_training)

    # train the enroller
    t = tempfile.mkstemp('ubm.hdf5', prefix='frltest_')[1]
    tool.train_enroller(facereclib.utils.tests.random_training_set(feature.shape, count=5, minimum=-5., maximum=5.), t)
    # assure that it is identical to the normal UBM projector
    tool.load_enroller(self.reference_dir('gmm_projector.hdf5'))

    # enroll model with the unprojected feature
    model = tool.enroll([feature])
    reference_model = tool.read_model(self.reference_dir('gmm_model.hdf5'))
    self.assertTrue(model.is_similar_to(reference_model))

    # score with unprojected feature and compare to the weird reference score ...
    probe = tool.read_probe(self.input_dir('dct_blocks.hdf5'))
    sim = tool.score(reference_model, probe)

    self.assertAlmostEqual(sim, 0.143875716)


  def test06b_gmm_video(self):
    # assure that the config file is readable
    tool = self.config('ubm_gmm_video')
    self.assertTrue(isinstance(tool, facereclib.tools.UBMGMMVideo))
    raise SkipTest("This test is not yet implemented")


  def test07_isv(self):
    # read input
    feature = bob.io.load(self.input_dir('dct_blocks.hdf5'))
    # assure that the config file is readable
    tool = self.config('isv')
    self.assertTrue(isinstance(tool, facereclib.tools.ISV))

    # Here, we use a reduced complexity for test purposes
    tool = facereclib.tools.ISV(
        number_of_gaussians = 2,
        subspace_dimension_of_u = 160,
        k_means_training_iterations = 1,
        gmm_training_iterations = 1,
        isv_training_iterations = 1,
        INIT_SEED = seed_value
    )
    self.assertTrue(tool.performs_projection)
    self.assertTrue(tool.requires_projector_training)
    self.assertTrue(tool.use_projected_features_for_enrollment)
    self.assertTrue(tool.split_training_features_by_client)
    self.assertFalse(tool.requires_enroller_training)

    # train the projector
    t = tempfile.mkstemp('ubm.hdf5', prefix='frltest_')[1]
    tool.train_projector(facereclib.utils.tests.random_training_set_by_id(feature.shape, count=5, minimum=-5., maximum=5.), t)
    if regenerate_refs:
      import shutil
      shutil.copy2(t, self.reference_dir('isv_projector.hdf5'))

    # load the projector file
    tool.load_projector(self.reference_dir('isv_projector.hdf5'))

    # compare ISV projector with reference
    hdf5file = bob.io.HDF5File(t)
    hdf5file.cd('Projector')
    projector_reference = bob.machine.GMMMachine(hdf5file)
    self.assertTrue(tool.m_ubm.is_similar_to(projector_reference))

    # compare ISV enroller with reference
    hdf5file.cd('/')
    hdf5file.cd('Enroller')
    enroller_reference = bob.machine.ISVBase(hdf5file)
    enroller_reference.ubm = projector_reference
    self.assertTrue(tool.m_isvbase.is_similar_to(enroller_reference))
    os.remove(t)

    # project the feature
    projected = tool.project(feature)
    if regenerate_refs:
      tool.save_feature(projected, self.reference_dir('isv_feature.hdf5'))

    # compare the projected feature with the reference
    projected_reference = tool.read_feature(self.reference_dir('isv_feature.hdf5'))
    self.assertTrue(projected[0].is_similar_to(projected_reference))

    # enroll model with the projected feature
    model = tool.enroll([projected[0]])
    if regenerate_refs:
      model.save(bob.io.HDF5File(self.reference_dir('isv_model.hdf5'), 'w'))
    reference_model = tool.read_model(self.reference_dir('isv_model.hdf5'))
    # compare the ISV model with the reference
    self.assertTrue(model.is_similar_to(reference_model))

    # check that the read_probe function reads the correct values
    probe = tool.read_probe(self.reference_dir('isv_feature.hdf5'))
    self.assertTrue(probe[0].is_similar_to(projected[0]))
    self.assertEqual(probe[1].any(), projected[1].any())

    # score with projected feature and compare to the weird reference score ...
    sim = tool.score(model, probe)
    self.assertAlmostEqual(sim, 0.002738819739893934)

    # score with a concatenation of the probe
    self.assertAlmostEqual(tool.score_for_multiple_probes(model, [probe, probe]), sim, places=5)


  def test07a_isv_video(self):
    # assure that the config file is readable
    tool = self.config('isv_video')
    self.assertTrue(isinstance(tool, facereclib.tools.ISVVideo))
    raise SkipTest("This test is not yet implemented")


  def test08_jfa(self):
    # NOTE: This test will fail when it is run solely. Please always run all Tool tests in order to assure that they work.
    # read input
    feature = bob.io.load(self.input_dir('dct_blocks.hdf5'))
    # assure that the config file is readable
    tool = self.config('jfa')
    self.assertTrue(isinstance(tool, facereclib.tools.JFA))

    # here, we use a reduced complexity for test purposes
    tool = facereclib.tools.JFA(
        number_of_gaussians = 2,
        subspace_dimension_of_u = 2,
        subspace_dimension_of_v = 2,
        k_means_training_iterations = 1,
        gmm_training_iterations = 1,
        jfa_training_iterations = 1,
        INIT_SEED = seed_value
    )
    self.assertTrue(tool.performs_projection)
    self.assertTrue(tool.requires_projector_training)
    self.assertTrue(tool.use_projected_features_for_enrollment)
    self.assertFalse(tool.split_training_features_by_client)
    self.assertTrue(tool.requires_enroller_training)

    # train the projector
    t = tempfile.mkstemp('ubm.hdf5', prefix='frltest_')[1]
    tool.train_projector(facereclib.utils.tests.random_training_set(feature.shape, count=5, minimum=-5., maximum=5.), t)
    if regenerate_refs:
      import shutil
      shutil.copy2(t, self.reference_dir('jfa_projector.hdf5'))

    # load the projector file
    tool.load_projector(self.reference_dir('jfa_projector.hdf5'))
    # compare JFA projector with reference
    new_machine = bob.machine.GMMMachine(bob.io.HDF5File(t))
    self.assertTrue(tool.m_ubm.is_similar_to(new_machine))
    os.remove(t)

    # project the feature
    projected = tool.project(feature)
    if regenerate_refs:
      projected.save(bob.io.HDF5File(self.reference_dir('jfa_feature.hdf5'), 'w'))
    # compare the projected feature with the reference
    projected_reference = tool.read_feature(self.reference_dir('jfa_feature.hdf5'))
    self.assertTrue(projected.is_similar_to(projected_reference))

    # train the enroller
    t = tempfile.mkstemp('enroll.hdf5', prefix='frltest_')[1]
    tool.train_enroller(self.train_gmm_stats(self.reference_dir('jfa_feature.hdf5'), count=5, minimum=-5., maximum=5.), t)
    if regenerate_refs:
      import shutil
      shutil.copy2(t, self.reference_dir('jfa_enroller.hdf5'))
    tool.load_enroller(self.reference_dir('jfa_enroller.hdf5'))
    # compare JFA enroller with reference
    enroller_reference = bob.machine.JFABase(bob.io.HDF5File(t))
    enroller_reference.ubm = new_machine
    self.assertTrue(tool.m_jfabase.is_similar_to(enroller_reference))
    os.remove(t)

    # enroll model with the projected feature
    model = tool.enroll([projected])
    if regenerate_refs:
      model.save(bob.io.HDF5File(self.reference_dir('jfa_model.hdf5'), 'w'))
    # assert that the model is ok
    reference_model = tool.read_model(self.reference_dir('jfa_model.hdf5'))
    self.assertTrue(model.is_similar_to(reference_model))

    # check that the read_probe function reads the requested data
    probe = tool.read_probe(self.reference_dir('jfa_feature.hdf5'))
    self.assertTrue(probe.is_similar_to(projected))

    # score with projected feature and compare to the weird reference score ...
    sim = tool.score(model, probe)
    self.assertAlmostEqual(sim, 0.25481840615836454)
    # score with a concatenation of the probe
    # self.assertAlmostEqual(tool.score_for_multiple_probes(model, [probe, probe]), sim)


  def test09_plda(self):
    # read input
    feature = bob.io.load(self.input_dir('linearize.hdf5'))
    # assure that the config file is readable
    tool = self.config('pca+plda')
    self.assertTrue(isinstance(tool, facereclib.tools.PLDA))

    # here, we use a reduced complexity for test purposes
    tool = facereclib.tools.PLDA(
        subspace_dimension_of_f = 2,
        subspace_dimension_of_g = 2,
        subspace_dimension_pca = 10,
        plda_training_iterations = 1,
        INIT_SEED = seed_value,
    )
    self.assertFalse(tool.performs_projection)
    self.assertTrue(tool.requires_enroller_training)

    # train the projector
    t = tempfile.mkstemp('pca+plda.hdf5', prefix='frltest_')[1]
    tool.train_enroller(facereclib.utils.tests.random_training_set_by_id(feature.shape, count=20, minimum=0., maximum=255.), t)
    if regenerate_refs:
      import shutil
      shutil.copy2(t, self.reference_dir('pca+plda_enroller.hdf5'))

    # load the projector file
    tool.load_enroller(self.reference_dir('pca+plda_enroller.hdf5'))
    # compare the resulting machines
    test_file = bob.io.HDF5File(t)
    test_file.cd('/pca')
    pca_machine = bob.machine.LinearMachine(test_file)
    test_file.cd('/plda')
    plda_machine = bob.machine.PLDABase(test_file)
    # TODO: compare the PCA machines
    #self.assertEqual(pca_machine, tool.m_pca_machine)
    # TODO: compare the PLDA machines
    #self.assertEqual(plda_machine, tool.m_plda_base_machine)
    os.remove(t)

    # enroll model
    model = tool.enroll([feature])
    if regenerate_refs:
      model.save(bob.io.HDF5File(self.reference_dir('pca+plda_model.hdf5'), 'w'))
    # TODO: compare the models with the reference
    #reference_model = tool.read_model(self.reference_dir('pca+plda_model.hdf5'))
    #self.assertEqual(model, reference_model)

    # score
    sim = tool.score(model, feature)
    self.assertAlmostEqual(sim, 0.)
    # score with a concatenation of the probe
    self.assertAlmostEqual(tool.score_for_multiple_probes(model, [feature, feature]), 0.)


  def test10_ivector(self):
    # NOTE: This test will fail when it is run solely. Please always run all Tool tests in order to assure that they work.
    # read input
    feature = bob.io.load(self.input_dir('dct_blocks.hdf5'))
    # assure that the config file is readable
    tool = self.config('ivector')
    self.assertTrue(isinstance(tool, facereclib.tools.IVector))

    # here, we use a reduced complexity for test purposes
    tool = facereclib.tools.IVector(
        number_of_gaussians = 2,
        subspace_dimension_of_t=2,       # T subspace dimension
        update_sigma = False, # TODO Do another test with True
        tv_training_iterations = 1,  # Number of EM iterations for the JFA training
        variance_threshold = 1e-5,
        INIT_SEED = seed_value
    )
    self.assertTrue(tool.performs_projection)
    self.assertTrue(tool.requires_projector_training)
    self.assertTrue(tool.use_projected_features_for_enrollment)
    self.assertFalse(tool.split_training_features_by_client)
    self.assertFalse(tool.requires_enroller_training)

    # train the projector
    t = tempfile.mkstemp('ubm.hdf5', prefix='frltest_')[1]
    tool.train_projector(facereclib.utils.tests.random_training_set(feature.shape, count=5, minimum=-5., maximum=5.), t)
    if regenerate_refs:
      import shutil
      shutil.copy2(t, self.reference_dir('ivector_projector.hdf5'))

    # load the projector file
    tool.load_projector(self.reference_dir('ivector_projector.hdf5'))

    # compare ISV projector with reference
    hdf5file = bob.io.HDF5File(t)
    hdf5file.cd('Projector')
    projector_reference = bob.machine.GMMMachine(hdf5file)
    self.assertTrue(tool.m_ubm.is_similar_to(projector_reference))

    # compare ISV enroller with reference
    hdf5file.cd('/')
    hdf5file.cd('Enroller')
    enroller_reference = bob.machine.IVectorMachine(hdf5file)
    enroller_reference.ubm = projector_reference
    self.assertTrue(tool.m_tv.is_similar_to(enroller_reference))
    os.remove(t)

    # project the feature
    projected = tool.project(feature)
    if regenerate_refs:
      tool.save_feature(projected, self.reference_dir('ivector_feature.hdf5'))

    # compare the projected feature with the reference
    projected_reference = tool.read_feature(self.reference_dir('ivector_feature.hdf5'))
    self.assertTrue(projected[0].is_similar_to(projected_reference))

    # enroll model with the projected feature
    # This is not yet supported
    # model = tool.enroll([projected[0]])
    # if regenerate_refs:
    #  model.save(bob.io.HDF5File(self.reference_dir('ivector_model.hdf5'), 'w'))
    #reference_model = tool.read_model(self.reference_dir('ivector_model.hdf5'))
    # compare the IVector model with the reference
    #self.assertTrue(model.is_similar_to(reference_model))

    # check that the read_probe function reads the correct values
    probe = tool.read_probe(self.reference_dir('ivector_feature.hdf5'))
    self.assertTrue(probe[0].is_similar_to(projected[0]))
    self.assertEqual(probe[1].any(), projected[1].any())

    # score with projected feature and compare to the weird reference score ...
    # This in not implemented yet

    # score with a concatenation of the probe
    # This is not implemented yet



