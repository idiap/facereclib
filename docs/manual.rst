.. vim: set fileencoding=utf-8 :
.. Manuel Guenther <Manuel.Guenther@idiap.ch>
.. Mon 23 04 2012

============
 FaceRecLib
============

The FaceRecLib toolkit is designed to run (face) verification/recognition experiments with the SGE grid infrastructure at Idiap. It is designed in a way that it should be easily possible to execute experiments combining different mixtures of:

* Image databases and their according protocols
* Image preprocessing
* Feature extraction 
* Recognition/Verification tools

In any case, results of these experiments will directly be comparable when the same database is employed.

Installation instructions
-------------------------

To install the FaceRecLib, please check the latest version of it via:

.. code-block:: sh

  $ git clone /idiap/group/torch5spro/sandboxes/facereclib2.git
  $ cd facereclib2

For the facereclib to work, it requires `Bob`_ to be installed. At Idiap, you can either have your local Bob installation or use the global one located at:

::

  > /idiap/group/torch5spro/nightlies/last/install/<VERSION>-release

where <VERSION> is your operating system version.

The facereclib project is based on the `BuildOut`_ python linking system. If you want to use another version of Bob than the nightlies, you have to modify the delivered *buildout.cfg* by specifying the path to your Bob installation. 

Afterwards, execute the buildout script by typing:

.. code-block:: sh
  
  $ /idiap/group/torch5spro/nightlies/externals/v2/linux-x86_64/bin/python2.6 bootstrap.py
  $ bin/buildout



Running experiments
-------------------

These two commands will automatically download all desired packages (`local.bob.recipe`_ and `gridtk`_) from GitHub and generate some scripts in the bin directory, including the script *bin/faceverify_zt.py*. This script can be used to employ face verification experiments. To use it you have to specify at least three command line parameters (see also the ``--help`` option):

* ``--database``: The configuration file for the database
* ``--features-extraction``: The configuration file for image preprocessing and feature extraction
* ``--tool-chain``: The configuration file for the face verification tool chain
  

If you want to run the experiments in the Idiap GRID, you simply can specify:

* ``--grid``: The configuration file for the grid setup.
  
If no grid configuration file is specified, the experiment is run sequentially on the local machine.

For several databases, feature types, recognition systems, and grid requirements the facereclib provides these configuration files. They are located in the *config/...* directories. None of the parameters in the configurations are fixed, so please feel free to test different settings.

Please note that not all combinations of features and tools make sense since the tools expect different kinds of features (e.g. UBM/GMM needs 2D features, whereas PCA expects 1D features).


By default, the verification result will be written to directory */idiap/user/$USER/<DATABASE>/<EXPERIMENT>/<SUBDIR>/<PROTOCOL>*, where

* DATABASE: the name of the database. It is read from the database configuration file
* EXPERIMENT: a user-specified experiment name (--sub-dir option), by default it is "default"
* SUBDIR: another user-specified name (--score-sub-dir), e.g. to specify different options of the experiment
* PROTOCOL: the protocol which is read from the database configuration file

After running a  ZT-Norm based experiment, the output directory contains two sub-directories *nonorm*, *ztnorm*, each of which contain the files *scores-dev* and *scores-eval*. One way to compute the final result is to use the *bob_compute_perf.py* script from your Bob installation, e.g., by calling:

.. code-block:: sh

  cd /idiap/user/$USER/<DATABASE>/<EXPERIMENT>/<SUBDIR>/<PROTOCOL>
  bob_compute_perf.py -d nonorm/scores-dev -t nonorm/scores-eval



Experiment design
-----------------

To be very flexible, the tool chain in the FaceRecLib is designed in several stages:

1. Image Preprocessing
2. Feature Extraction
3. Feature Projection
4. Model Enrollment
5. Scoring

Note that not all tools implement all of the stages. 

Image Preprocessing
~~~~~~~~~~~~~~~~~~~
In the image preprocessing step, the image usually will be aligned to the eye positions. Currently, the eye positions are expected to be read from file, but later versions of the image preprocessing might also perform face detection. Currently, there are two versions of image preprocessing:

* Alignment of the image
* Alignment of the image + Tan-Triggs


Feature Extraction
~~~~~~~~~~~~~~~~~~
If required by the feature extraction tool, an optional feature extraction training using all preprocessed images of the training set is performed. In the feature extraction stage, the features from all images in the database are extracted and stored. Currently, these different feature types are implemented:

* Eigenfaces (require training)
* DCT Blocks
* (Extended) Local Gabor Binary Pattern Histogram Sequences (E)LGBPHS
* Gabor grid graphs including Gabor jets with or w/o Gabor phases


Feature Projection
~~~~~~~~~~~~~~~~~~
Some provided tools need to process the features before they can be used for verification. In the FaceRecLib, this step is referenced as the **projection** step. Again, the projection might require training, which is executed using the extracted features from the training set. Afterwards, all features are projected (using the the previously trained Projector).


Model Enrollment
~~~~~~~~~~~~~~~~
Model enrollment defines the stage, where several (projected or unprojected) features of one identity are used to enroll the model for that identity. In the easiest case, the features are simply averaged, and the average feature is used as a model. More complex procedures, which again might require a model enrollment training stage, create models in a different way.


Scoring
~~~~~~~
In the final scoring stage, the models are compared to probe features and a similarity score is computed for each pair of model and probe. Some of the models (the so-called T-Norm-Model) and some of the probe features (so-called Z-Norm-probe-features) are split up, so they can be used to normalize the scores later on.



Command line options
--------------------
Additionally to the required command line options discussed above, there are several options to modify the behavior of the FaceRecLib experiments. One set of command line options change the directory structure of the output:

* ``--temp-directory``: Base directory where to write temporary files into (the default is */idiap/temp/$USER/<DATABASE>* when using the grid or */scratch/$USER/<DATABASE>* when executing jobs locally)
* ``--user-directory``: Base directory where to write the results
* ``--sub-directory``: sub-directory into *<TEMP_DIR>* and *<USER_DIR>* where the files generated by the experiment will be put
* ``--score-sub-directory``: name of the sub-directory in *<USER_DIR>/<PROTOCOL>* where the scores are put into
  
If you want to re-use parts previous experiments, you can specify the directories (which are relative to the *<TEMP_DIR>*, but you can also specify absolute paths):

* ``--preprocessed-image-directory``
* ``--features-directory``
* ``--projected-directory``
* ``--models-directories`` (one for each the Models and the T-Norm-Models)

or even trained Extractor, Projector, or Enroler (i.e., the results of the extraction, projection, or enrollment training):

* ``--extractor-file``
* ``--projector-file``
* ``--enroler-file``

For that purpose, it is also useful to skip parts of the tool chain. To do that you can use:

* ``--skip-preprocessing``
* ``--skip-feature-extraction-training``
* ``--skip-feature-extraction``
* ``--skip-projection-training``
* ``--skip-projection``
* ``--skip-enroler-training``
* ``--skip-model-enrolment``
* ``--skip-score-computation``
  
although by default files that already exist are not re-created. To enforce the re-creation of the files, you can use the ``--force`` option, which of course can be combined with the ``--skip...``-options (in which case the skip is preferred).

There are some more command line options that can be specified:

* ``--no-zt-norm``: Disables the computation of the ZT-Norm scores
* ``--preload-probes``: Speeds up the score computation by loading all probe features (by default, they are loaded each time they are needed). Use this option only, when you are sure that all probe features fit into memory.
* ``--dry-run``: When the grid is enabled, only print the tasks that would have been sent to the grid without actually send them. **WARNING** This command line option is ignored when no ``--grid`` option was specified!


The GBU database
----------------
There is another script *bin/faceverify_gbu.py* that executes experiments on the Good, Bad, and Ugly (GBU) database. In principle, most of the parameters from above can be used. One violation is that instead of the ``--database`` option, now the ``--database-directory`` (the directory containing the GBU database files, normally: *config/database*) needs to be specified.

When running experiments on the GBU database, the default GBU protocol (as provided by `NIST`_) is used. Hence, training is performed on the special Training set, and experiments are executed using the Target set as models (using a single image for model enrollment) and the Query set as probe.

The GBU protocol does not specify T-Norm-models or Z-Norm-probes, nor it splits off development and test set. Hence, only a single score file is generated, which might later on be converted into an ROC curve using Bob functions.

.. _Bob: http://idiap.github.com/bob/
.. _local.bob.recipe: https://github.com/idiap/local.bob.recipe
.. _gridtk: https://github.com/idiap/gridtk
.. _BuildOut: http://www.buildout.org/
.. _NIST: http://www.nist.gov/itl/iad/ig/focs.cfm

