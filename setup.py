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
        'faceverify.py = facereclib.script.faceverify:main'
        ],
      },

    #long_description=open('doc/install.rst').read(),

    install_requires=[
        "argparse", # better option parsing
        "gridtk",   # SGE job submission at Idiap
        #"bob",      # base signal proc./machine learning library
    ],
)
