The Face Recognition Library
============================

Welcome to the Face Recognition Library.
This library is designed to perform a fair comparison of face recognition algorithms.
It contains scripts to execute various kinds of face recognition experiments on a variety of facial image databases.

About
-----

This library is developed at the `Biometrics group <http://www.idiap.ch/~marcel/professional/Research_Team.html>`_ at the `Idiap Research Institute <http://www.idiap.ch>`_.
The FaceRecLib is designed to run face recognition experiments in a comparable and reproducible manner.
To achieve this goal, interfaces to many publicly available facial image databases are contained, and default evaluation protocols are defined, e.g.:

- Face Recognition Grand Challenge version 2
- The Good, The Bad and the Ugly
- Labeled Faces in the Wild
- Multi-PIE
- SCface
- MOBIO
- BANCA
- CAS-PEAL
- AR face database
- XM2VTS
- The AT&T database of faces (formerly known as ORL)

Additionally, a broad variety of state-of-the-art face recognition algorithms such as:

- Linear Discriminant Analysis
- Probabilistic Linear Discriminant Analysis
- Local Gabor Binary Pattern Histogram Sequences
- Graph Matching
- Inter-Session Variability Modeling
- Bayesian Intrapersonal/Extrapersonal Classifier

is provided, and running baseline algorithms is as easy as going to the command line and typing::

  $ bin/baselines.py --database frgc --algorithm lda

Furthermore, tools to evaluate the results can easily be used to create scientific plots.

Interested? You are highly welcome to try it out!


Installation
------------

We proudly present the first version of the FaceRecLib on pypi.
To download the FaceRecLib, please go to http://pypi.python.org/pypi/facereclib, click on the download button and extract the .zip file to a folder of your choice.

Bob
...

The FaceRecLib is a satellite package of the free signal processing and machine learning library Bob_.
You will need a copy of Bob in version 1.2.0 it to run the algorithms.
Please download Bob_ from its webpage.

.. note::
  At Idiap_, Bob_ is globally installed.
  This version of the FaceRecLib is bound to Bob version 1.2.0, which does not correspond to the one installed.
  However, the correct version of Bob is marked in the buildout.cfg.

After downloading, you should go to the console and write::

  $ python bootstrap.py
  $ bin/buildout
  $ bin/sphinx-build docs sphinx

This will download all required packages and install them locally.
If you don't want all the database packages to be downloaded, please remove the xbob.db.[database] lines from the ``eggs`` section of the file **buildout.cfg** in the main directory before calling the three commands above.

Now, you can open the documentation by typing::

  $ firefox sphinx/index.html

and read further instructions on how to use this library.

.. _bob: http://www.idiap.ch/software/bob
.. _idiap: http://www.idiap.ch
.. _bioidiap at github: http://www.github.com/bioidiap
