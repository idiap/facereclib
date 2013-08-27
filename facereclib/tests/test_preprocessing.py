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
import facereclib
import bob
from nose.plugins.skip import SkipTest

import pkg_resources

regenerate_refs = False

class PreprocessingTest(unittest.TestCase):

  def input_dir(self, file):
    return pkg_resources.resource_filename('facereclib.tests', file)

  def reference_dir(self, file):
    ref = pkg_resources.resource_filename('facereclib.tests', os.path.join('preprocessing', file))
    facereclib.utils.ensure_dir(os.path.dirname(ref))
    return ref

  def input(self):
    return (bob.io.load(self.input_dir("testimage.jpg")), facereclib.utils.read_annotations(self.input_dir("testimage.pos"), 'named'))

  def config(self, resource):
    return facereclib.utils.tests.configuration_file(resource, 'preprocessor', 'preprocessing')

  def execute(self, preprocessor, data, annotations, reference):
    # execute the preprocessor
    preprocessed = preprocessor(data, annotations)
    if regenerate_refs:
      bob.io.save(preprocessed, self.reference_dir(reference))

    self.assertTrue((numpy.abs(bob.io.load(self.reference_dir(reference)) - preprocessed) < 1e-5).all())



  def test00_null_preprocessor(self):
    # read input
    data, annotation = self.input()
    # the null preprocessor currently has no config file
    preprocessor = facereclib.preprocessing.NullPreprocessor()
    # execute preprocessor
    self.execute(preprocessor, data, annotation, 'gray.hdf5')


  def test01_face_crop(self):
    # read input
    data, annotation = self.input()
    preprocessor = self.config('face-crop')
    # execute face cropper
    self.execute(preprocessor, data, annotation, 'cropped.hdf5')

    # test the preprocessor with fixed eye positions
    # here, we read a special config file that sets the same fixed eye positions as given in the test data
    preprocessor = facereclib.utils.resources.load_resource(self.reference_dir('face_crop_fixed.py'), 'preprocessor')
    # execute face cropper;
    # result must be identical to the original face cropper (same eyes are used)
    self.execute(preprocessor, data, None, 'cropped.hdf5')

    # test the preprocessor with offset
    preprocessor = self.config('face_crop_with_offset')
    preprocessed = preprocessor(data, annotation)
    # results of the inner parts must be similar
    self.assertTrue((numpy.abs(bob.io.load(self.reference_dir('cropped.hdf5')) - preprocessed[2:-2, 2:-2]) < 1e-10).all())


  def test02_tan_triggs(self):
    # read input
    data, annotation = self.input()
    preprocessor = self.config('tan-triggs')
    # execute preprocessor
    self.execute(preprocessor, data, annotation, 'tan_triggs_cropped.hdf5')

    # test if the preprocessor with offset at least loads
    preprocessor = self.config('tan_triggs_with_offset')
    self.assertTrue(isinstance(preprocessor, facereclib.preprocessing.TanTriggs))

    # execute the preprocessor without cropping
    preprocessor = facereclib.preprocessing.TanTriggs()
    self.execute(preprocessor, data, None, 'tan_triggs.hdf5')


  def test02a_tan_triggs_video(self):
    preprocessor = self.config('tan_triggs_video')
    self.assertTrue(isinstance(preprocessor, facereclib.preprocessing.TanTriggsVideo))
    raise SkipTest("Video tests are currently skipped.")
    # read input
    f = '/idiap/home/rwallace/work/databases/banca-video/output/frames/1024_f_g2_s11_1024_en_4.hdf5'
    if not os.path.exists(f):
      raise SkipTest("The original video '%s' for the test is not available."%f)

    # read the original video using the preprocessor
    original = preprocessor.read_original_data(f)

    # preprocess
    preprocessed = preprocessor(original)
    if regenerate_refs:
      preprocessed.save(bob.io.HDF5File(self.reference_dir('video.hdf5'), 'w'))

    reference = preprocessor.read_data(self.reference_dir('video.hdf5'))
    self.assertEqual(preprocessed, reference)


  def test03_self_quotient(self):
    # read input
    data, annotation = self.input()
    preprocessor = self.config('self-quotient')
    # execute preprocessor
    self.execute(preprocessor, data, annotation, 'self_quotient_cropped.hdf5')
#    self.execute(preprocessor, data, None, 'self_quotient.hdf5')


  def test04_inorm_lbp(self):
    # read input
    data, annotation = self.input()
    preprocessor = self.config('inorm-lbp')
    # execute preprocessor
    self.execute(preprocessor, data, annotation, 'inorm_cropped.hdf5')
#    self.execute(preprocessor, data, None, 'inorm.hdf5')


  def test05_histogram(self):
    # read input
    data, annotation = self.input()
    preprocessor = self.config('histogram-equalize')
    # execute preprocessor
    self.execute(preprocessor, data, annotation, 'histogram_cropped.hdf5')
#    self.execute(preprocessor, data, None, 'histogram.hdf5')


  def test06a_key_points(self):
    # read input
    data, annotation = self.input()
    preprocessor = self.config('keypoints')

    # execute preprocessor
    preprocessed = preprocessor(data, annotation)
    if regenerate_refs:
      preprocessor.save_data(preprocessed, self.reference_dir('key_points.hdf5'))

    reference = preprocessor.read_data(self.reference_dir('key_points.hdf5'))
    # check if it is near the reference data and positions
    data, annots = preprocessed
    data2, annot2 = reference
    self.assertTrue((numpy.abs(data - data2) < 1e-5).all())
    self.assertTrue((annots == annot2).all())

  def test06b_key_points(self):
    # read input
    data, annotation = self.input()
    preprocessor = self.config('keypoints_lfw')

    # execute preprocessor
    preprocessed = preprocessor(data, annotation)
    if regenerate_refs:
      preprocessor.save_data(preprocessed, self.reference_dir('key_points_cropped.hdf5'))

    reference = preprocessor.read_data(self.reference_dir('key_points_cropped.hdf5'))
    # check if it is near the reference data and positions
    data, annots = preprocessed
    data2, annot2 = reference
    self.assertTrue((numpy.abs(data - data2) < 1e-5).all())
    self.assertTrue((annots == annot2).all())
