from setuptools import setup, find_packages

setup(
    name='facereclib',
    version='0.2',
    description='Face recognition and face verification toolchain',

    #url='http://pypi.python.org/pypi/TowelStuff/',
    #license='LICENSE.txt',

    author='Manuel Guenther',
    author_email='Manuel.Guenther@idiap.ch',

    packages=find_packages(),

    entry_points={
      'console_scripts': [
        'faceverify.py = facereclib.script.faceverify:main',
#        'faceverify_gbu.py = facereclib.script.faceverify_gbu:main',
        'faceverify_lfw.py = facereclib.script.faceverify_lfw:main',
#        'faceverify_pose.py = facereclib.script.faceverify_pose:main',
#        'parameter_test.py = facereclib.script.parameter_test:main',
        'baselines.py = facereclib.script.baselines:main',
        'resources.py = facereclib.utils.resources:print_all_resources'
      ],

      'facereclib.database': [
        'atnt              = facereclib.configurations.databases.atnt_Default:database',
        'arface            = facereclib.configurations.databases.arface_all:database',
        'banca             = facereclib.configurations.databases.banca_P_color:database',
        'lfw               = facereclib.configurations.databases.lfw_view1:database',
        'mobio-male        = facereclib.configurations.databases.mobio_male:database',
        'mobio-female      = facereclib.configurations.databases.mobio_female:database',
        'multipie-P        = facereclib.configurations.databases.multipie_P:database',
        'multipie-U        = facereclib.configurations.databases.multipie_U:database',
        'scface            = facereclib.configurations.databases.scface_combined:database',
        'xm2vts            = facereclib.configurations.databases.xm2vts_lp1:database'
      ],

      'facereclib.preprocessor': [
        'face-crop         = facereclib.configurations.preprocessing.face_crop:preprocessor',
        'histogram-equalize= facereclib.configurations.preprocessing.histogram_equalize:preprocessor',
        'inorm-lbp         = facereclib.configurations.preprocessing.inorm_lbp:preprocessor',
        'self-quotient     = facereclib.configurations.preprocessing.self_quotient:preprocessor',
        'tan-triggs        = facereclib.configurations.preprocessing.tan_triggs:preprocessor'
      ],

      'facereclib.feature_extractor': [
        'dct               = facereclib.configurations.features.dct_blocks:feature_extractor',
        'eigenfaces        = facereclib.configurations.features.eigenfaces:feature_extractor',
        'grid-graph        = facereclib.configurations.features.grid_graph:feature_extractor',
        'lgbphs            = facereclib.configurations.features.lgbphs:feature_extractor',
        'linearize         = facereclib.configurations.features.linearize:feature_extractor',
        'sift              = facereclib.configurations.features.sift_keypoints:feature_extractor',
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
      ]
    },

    #long_description=open('doc/install.rst').read(),

    install_requires=[
      "setuptools", # for whatever
      "bob >= 1.1.1",      # base signal proc./machine learning library
    ],
)
