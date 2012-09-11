from setuptools import setup, find_packages

setup(
    name='facereclib',
    version='0.1',
    description='Face recognition and face verification toolchain',

    #url='http://pypi.python.org/pypi/TowelStuff/',
    #license='LICENSE.txt',

    author='Manuel Guenther',
    author_email='Manuel.Guenther@idiap.ch',

    packages=find_packages(),

    entry_points={
      'console_scripts': [
        'faceverify_zt.py = facereclib.script.faceverify_zt:main',
        'faceverify_gbu.py = facereclib.script.faceverify_gbu:main',
        'faceverify_lfw.py = facereclib.script.faceverify_lfw:main',
        'faceverify_pose.py = facereclib.script.faceverify_pose:main',
        'parameter_test.py = facereclib.script.parameter_test:main',
        'baselines.py = facereclib.script.baselines:main'
        ],
      },

    #long_description=open('doc/install.rst').read(),

    install_requires=[
        "setuptools", # for whatever
        "gridtk",   # SGE job submission at Idiap
        "bob >= 1.1.0a0",      # base signal proc./machine learning library
        # databases
        "xbob.db.arface",
        "xbob.db.atnt",
        "xbob.db.banca",
        "xbob.db.frgc",
        "xbob.db.lfw",
        "xbob.db.gbu",
        "xbob.db.mobio",
        "xbob.db.multipie",
        "xbob.db.scface",
        "xbob.db.xm2vts"
    ],
)
