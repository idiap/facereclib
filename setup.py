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

    #url='http://pypi.python.org/pypi/TowelStuff/',
    #license='LICENSE.txt',

    author='Manuel Guenther',
    author_email='Manuel.Guenther@idiap.ch',

    packages=find_packages(),

    entry_points={
      'console_scripts': [
        'faceverify.py = facereclib.script.faceverify:main',
        'faceverify_gbu.py = facereclib.script.faceverify_gbu:main',
        'faceverify_lfw.py = facereclib.script.faceverify_lfw:main',
        'isv_trainer.py = facereclib.script.isv_trainer:main',
#        'faceverify_pose.py = facereclib.script.faceverify_pose:main',
        'parameter_test.py = facereclib.script.parameter_test:main',
        'baselines.py = facereclib.script.baselines:main',
        'resources.py = facereclib.utils.resources:print_all_resources'
      ],

      'facereclib.database': [
        'arface            = facereclib.configurations.databases.arface_all:database',
        'atnt              = facereclib.configurations.databases.atnt_Default:database',
        'banca             = facereclib.configurations.databases.banca_P:database',
        'caspeal           = facereclib.configurations.databases.caspeal_lighting:database',
        'frgc              = facereclib.configurations.databases.frgc_201:database',
        'gbu               = facereclib.configurations.databases.gbu_Good:database',
        'lfw               = facereclib.configurations.databases.lfw_view1_unrestricted:database',
        'mobio             = facereclib.configurations.databases.mobio_male:database',
        'multipie          = facereclib.configurations.databases.multipie_U:database',
        'scface            = facereclib.configurations.databases.scface_combined:database',
        'xm2vts            = facereclib.configurations.databases.xm2vts_lp1:database',
        'audio_banca_p     = facereclib.configurations.databases.audio_banca_P:database',
        'audio_banca_g     = facereclib.configurations.databases.audio_banca_G:database',
      ],

      'facereclib.preprocessor': [
        'face-crop         = facereclib.configurations.preprocessing.face_crop:preprocessor',
        'histogram-equalize= facereclib.configurations.preprocessing.histogram_equalize:preprocessor',
        'inorm-lbp         = facereclib.configurations.preprocessing.inorm_lbp:preprocessor',
        'self-quotient     = facereclib.configurations.preprocessing.self_quotient:preprocessor',
        'tan-triggs        = facereclib.configurations.preprocessing.tan_triggs:preprocessor',
        'audio-preprocessor= facereclib.configurations.preprocessing.audio_preprocessor:preprocessor',
      ],

      'facereclib.feature_extractor': [
        'dct               = facereclib.configurations.features.dct_blocks:feature_extractor',
        'eigenfaces        = facereclib.configurations.features.eigenfaces:feature_extractor',
        'grid-graph        = facereclib.configurations.features.grid_graph:feature_extractor',
        'lgbphs            = facereclib.configurations.features.lgbphs:feature_extractor',
        'linearize         = facereclib.configurations.features.linearize:feature_extractor',
        'sift              = facereclib.configurations.features.sift_keypoints:feature_extractor',
        'cepstral          = facereclib.configurations.features.cepstral:feature_extractor',
      ],

      'facereclib.tool': [
        'bic               = facereclib.configurations.tools.bic:tool',
        'gabor-jet         = facereclib.configurations.tools.gabor_jet:tool',
        'isv               = facereclib.configurations.tools.isv:tool',
        'jfa               = facereclib.configurations.tools.jfa:tool',
        'lda               = facereclib.configurations.tools.lda:tool',
        'pca+lda           = facereclib.configurations.tools.pca_lda:tool',
        'lgbphs            = facereclib.configurations.tools.lgbphs:tool',
        'pca               = facereclib.configurations.tools.pca:tool',
        'plda              = facereclib.configurations.tools.plda:tool',
        'pca+plda          = facereclib.configurations.tools.pca_plda:tool',
        'gmm               = facereclib.configurations.tools.ubm_gmm:tool',
      ],

      'facereclib.grid': [
        'grid              = facereclib.configurations.grid.grid',
        'demanding         = facereclib.configurations.grid.demanding',
        'very-demanding    = facereclib.configurations.grid.very_demanding',
        'gbu               = facereclib.configurations.grid.gbu',
        'lfw               = facereclib.configurations.grid.lfw',
        'small             = facereclib.configurations.grid.small',
        'isv               = facereclib.configurations.grid.isv_training',
      ]
    },

    #long_description=open('doc/install.rst').read(),

    install_requires=[
      "setuptools", # for whatever
      "bob >= 1.1.1",      # base signal proc./machine learning library
      "xbob.db.atnt",      # for test purposes, the (freely available) AT&T database is required
      "pysox",
    ],
)
