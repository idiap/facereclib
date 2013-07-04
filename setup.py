#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Tue Oct 30 09:53:56 CET 2012
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


from setuptools import setup, find_packages

setup(
    name='facereclib',
    version='1.0.0a1',
    description='Compare a variety of face recognition algorithms by running them on many image databases with default protocols.',

    url='https://github.com/bioidiap/facereclib',
    #license='LICENSE.txt',

    author='Manuel Guenther',
    author_email='manuel.guenther@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),

    entry_points={
      # console scripts that will be created in bin/
      'console_scripts': [
        'faceverify.py = facereclib.script.faceverify:main',
        'faceverify_gbu.py = facereclib.script.faceverify_gbu:main',
        'faceverify_lfw.py = facereclib.script.faceverify_lfw:main',
        'isv_trainer.py = facereclib.script.isv_trainer:main',
        'ivector_trainer.py = facereclib.script.ivector_trainer:main',
#        'faceverify_pose.py = facereclib.script.faceverify_pose:main',
        'parameter_test.py = facereclib.script.parameter_test:main',
        'baselines.py = facereclib.script.baselines:main',
        'resources.py = facereclib.utils.resources:print_all_resources',
        'collect_results.py = facereclib.script.collect_results:main'
      ],

      # registered database short cuts
      'facereclib.database': [
        'arface            = facereclib.configurations.databases.arface:database',
        'atnt              = facereclib.configurations.databases.atnt:database',
        'banca             = facereclib.configurations.databases.banca:database',
        'caspeal           = facereclib.configurations.databases.caspeal:database',
        'frgc              = facereclib.configurations.databases.frgc:database',
        'gbu               = facereclib.configurations.databases.gbu:database',
        'lfw               = facereclib.configurations.databases.lfw_unrestricted:database',
        'mobio             = facereclib.configurations.databases.mobio:database',
        'multipie          = facereclib.configurations.databases.multipie:database',
        'scface            = facereclib.configurations.databases.scface:database',
        'xm2vts            = facereclib.configurations.databases.xm2vts:database',
      ],

      # registered preprocessors
      'facereclib.preprocessor': [
        'face-crop         = facereclib.configurations.preprocessing.face_crop:preprocessor',
        'histogram-equalize= facereclib.configurations.preprocessing.histogram_equalize:preprocessor',
        'inorm-lbp         = facereclib.configurations.preprocessing.inorm_lbp:preprocessor',
        'self-quotient     = facereclib.configurations.preprocessing.self_quotient:preprocessor',
        'tan-triggs        = facereclib.configurations.preprocessing.tan_triggs:preprocessor',
        'audio-preprocessor= facereclib.configurations.preprocessing.audio_preprocessor:preprocessor',
      ],

      # registered feature extractors
      'facereclib.feature_extractor': [
        'dct               = facereclib.configurations.features.dct_blocks:feature_extractor',
        'eigenfaces        = facereclib.configurations.features.eigenfaces:feature_extractor',
        'grid-graph        = facereclib.configurations.features.grid_graph:feature_extractor',
        'lgbphs            = facereclib.configurations.features.lgbphs:feature_extractor',
        'linearize         = facereclib.configurations.features.linearize:feature_extractor',
        'sift              = facereclib.configurations.features.sift_keypoints:feature_extractor',
        'cepstral          = facereclib.configurations.features.cepstral:feature_extractor',
      ],

      # registered face recognition algorithms
      'facereclib.tool': [
        'bic               = facereclib.configurations.tools.bic:tool',
        'gabor-jet         = facereclib.configurations.tools.gabor_jet:tool',
        'isv               = facereclib.configurations.tools.isv:tool',
        'ivector           = facereclib.configurations.tools.ivector:tool',
        'jfa               = facereclib.configurations.tools.jfa:tool',
        'lda               = facereclib.configurations.tools.lda:tool',
        'pca+lda           = facereclib.configurations.tools.pca_lda:tool',
        'lgbphs            = facereclib.configurations.tools.lgbphs:tool',
        'pca               = facereclib.configurations.tools.pca:tool',
        'plda              = facereclib.configurations.tools.plda:tool',
        'pca+plda          = facereclib.configurations.tools.pca_plda:tool',
        'gmm               = facereclib.configurations.tools.ubm_gmm:tool',
      ],

      # registered SGE grid configuration files
      'facereclib.grid': [
        'grid              = facereclib.configurations.grid.grid',
        'demanding         = facereclib.configurations.grid.demanding',
        'very-demanding    = facereclib.configurations.grid.very_demanding',
        'gbu               = facereclib.configurations.grid.gbu',
        'lfw               = facereclib.configurations.grid.lfw',
        'small             = facereclib.configurations.grid.small',
        'isv               = facereclib.configurations.grid.isv_training',
        'ivector           = facereclib.configurations.grid.ivector_training',
      ],

      # registered tests (will, e.g., be run in the xbob.db.aggregator)
      'bob.test' : [
        'databases         = facereclib.tests.test_databases:DatabaseTest',
        'preprocessors     = facereclib.tests.test_preprocessing:PreprocessingTest',
        'feature_extractors= facereclib.tests.test_features:FeatureExtractionTest',
        'tools             = facereclib.tests.test_tools:ToolTest',
        'scripts           = facereclib.tests.test_scripts:ScriptTest'
      ]
    },

    #long_description=open('doc/install.rst').read(),

    install_requires=[
      "setuptools", # for whatever
      "bob >= 1.2.0a0",    # base signal proc./machine learning library
      "xbob.db.atnt",      # for test purposes, the (freely available) AT&T database is required
    ],
)
