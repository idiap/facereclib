.. vim: set fileencoding=utf-8 :
.. Laurent El Shafey <laurent.el-shafey@idiap.ch>
.. Wed 25 Jan 2012.

============
 FaceRecLib
============

The FaceRecLib toolkit is designed to run (face) verification/recognition
experiments, possibly with the SGE grid infrastructure (available at Idiap).

It requires the following softwares:

- bob     (/idiap/group/torch5spro/git/torch5spro.git)
- gridtk  (/idiap/group/torch5spro/sandboxes/gridtk.git)


Installation instructions
-------------------------

To get a copy of the FaceRecLib package, just clone its git repository:

.. code-block:: sh

  $ git clone /idiap/group/torch5spro/sandboxes/facereclib.git

Change directories to the root of the `facereclib` package and setup the required
links:

.. code-block:: sh

  $ cd facereclib
  $ ln -s GRIDTK_DIR . #link to gridtk root
  $ ln -s BOB_DIR .    #link to bob root


Running experiments
-------------------

The FaceRecLib toolkit relies on configuration files and comes with a shell.py script, 
which setups the required environment.

In the following, we describe the example of running DCT-GMM experiments for the protocol P
of the BANCA database.

The first thing to do is to update the output directories in the config/gmm_banca_P.py 
configuration file.

Then, to compute the DCT features, you just need to run:

.. code-block:: sh

  $ ./shell.py -- grid_make_features.py -s script/dctfeatures.py -c config/gmm_banca_P.py

If you do not have access to the (Idiap) SGE infrastructure, you will have to add 
the -j option:

.. code-block:: sh

  $ ./shell.py -- grid_make_features.py -s script/dctfeatures.py -c config/gmm_banca_P.py -j


Then, to use these features to run GMM experiments, you just have to launch:

.. code-block:: sh

  $ ./shell.py -- grid_gmm.py -c config/gmm_banca_P.py

This should train the Universal Background Model, enrol the client models, compute and normalize
the scores, and put them in two text files (4-columns formatted score file) scores-dev and 
scores-eval.

The bob library can then be used to compute HTER-like values.
