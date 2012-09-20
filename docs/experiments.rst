.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

Running specialized experiments
===============================

.. TODO::

  Check if all variables are still valid, and replace them if not.

If you want to run experiments with a different setup, you can choose to use the ``bin/faceverify.py`` script.
To use it you have to specify at least three command line parameters (see also the ``--help`` option):

* ``--database``: The configuration file for the database
* ``--preprocessing``: The configuration file for image preprocessing
* ``--features``: The configuration file for feature extraction
* ``--tool``: The configuration file for the face verification algorithm

If you want to run the experiments in the `Idiap`_ SGE grid, you simply can specify:

* ``--grid``: The configuration file for the grid setup.

If no grid configuration file is specified, the experiment is run sequentially on the local machine.
For several databases, feature types, recognition algorithms, and grid requirements the |project| provides these configuration files.
They are located in the **config/...** directories.
It is also save to design one experiment and re-use one configuration file for all options as long as the configuration file includes all desired information:

* The database: ``name, db, protocol; img_input_dir, img_input_ext``; optional: ``pos_input_dir, pos_input_ext, first_annot; all_files_option, world_extractor_options, world_projector_options, world_enroler_options, features_by_clients_options``
* The preprocessing: ``preprocessor = facereclib.preprocessing.<PREPROCESSOR>``; optional: ``color_channel``; plus configurations of the preprocessor itself
* The features: ``feature_extractor = facereclib.features.<FEATURE_EXTRACTOR>``; plus configurations of the features themselves
* The tool: ``tool = facereclib.tools.<TOOL>``; plus configurations of the tool itself
* Grid parameters: ``training_queue; number_of_images_per_job, preprocessing_queue; number_of_features_per_job, extraction_queue, number_of_projections_per_job, projection_queue; number_of_models_per_enrol_job, enrol_queue; number_of_models_per_score_job, score_queue``

None of the parameters in the configurations are fixed, so please feel free to test different settings.
Please note that not all combinations of features and tools make sense since the tools expect different kinds of features (e.g. UBM/GMM needs 2D features, whereas PCA expects 1D features).


By default, the verification result will be written to directory */idiap/user/$USER/<DATABASE>/<EXPERIMENT>/<SUBDIR>/<PROTOCOL>*, where

* DATABASE: the name of the database. It is read from the database configuration file
* EXPERIMENT: a user-specified experiment name (``--sub-dir`` option), by default it is ``default``
* SUBDIR: another user-specified name (``--score-sub-dir`` option), e.g. to specify different options of the experiment
* PROTOCOL: the protocol which is read from the database configuration file

After running a  ZT-Norm based experiment, the output directory contains two sub-directories *nonorm*, *ztnorm*, each of which contain the files *scores-dev* and *scores-eval*.
One way to compute the final result is to use the *bob_compute_perf.py* script from your Bob installation, e.g., by calling:

.. code-block:: sh

  $ cd /idiap/user/$USER/<DATABASE>/<EXPERIMENT>/<SUBDIR>/<PROTOCOL>
  $ bob_compute_perf.py -d nonorm/scores-dev -t nonorm/scores-eval


Temporary files will by default be put to */scratch/$USER/<DATABASE>/<EXPERIMENT>* or */idiap/temp/$USER/<DATABASE>/<EXPERIMENT>* when run locally or in the grid, respectively.


Experiment design
-----------------

.. TODO::

  Add and correct the current list of implemented tools

To be very flexible, the tool chain in the |project| is designed in several stages:

1. Image Preprocessing
2. Feature Extraction
3. Feature Projection
4. Model Enrollment
5. Scoring

Note that not all tools implement all of the stages.

Image Preprocessing
~~~~~~~~~~~~~~~~~~~
In the image preprocessing step, the image usually will be aligned to the eye positions.
Currently, the eye positions are expected to be read from file, but later versions of the image preprocessing might also perform face detection.
Currently, there are four versions of image preprocessing:

* Alignment of the image
* Alignment of the image + Tan-Triggs
* Alignment of the image + histogram equalization
* Alignment of the image + self quotient image
* Alignment of the image + i-norm LBP preprocessing



Feature Extraction
~~~~~~~~~~~~~~~~~~
If required by the feature extraction tool, an optional feature extraction training using all preprocessed images of the training set is performed.
In the feature extraction stage, the features from all images in the database are extracted and stored.
Currently, these different feature types are implemented:

* Pixel values
* Eigenfaces (require training)
* DCT Blocks
* (Extended) Local Gabor Binary Pattern Histogram Sequences (E)LGBPHS
* Gabor grid graphs including Gabor jets with or w/o Gabor phases


Feature Projection
~~~~~~~~~~~~~~~~~~
Some provided tools need to process the features before they can be used for verification.
In the |project|, this step is referenced as the **projection** step.
Again, the projection might require training, which is executed using the extracted features from the training set.
Afterward, all features are projected (using the the previously trained Projector).


Model Enrollment
~~~~~~~~~~~~~~~~
Model enrollment defines the stage, where several (projected or unprojected) features of one identity are used to enroll the model for that identity.
In the easiest case, the features are simply averaged, and the average feature is used as a model.
More complex procedures, which again might require a model enrollment training stage, create models in a different way.


Scoring
~~~~~~~
In the final scoring stage, the models are compared to probe features and a similarity score is computed for each pair of model and probe.
Some of the models (the so-called T-Norm-Model) and some of the probe features (so-called Z-Norm-probe-features) are split up, so they can be used to normalize the scores later on.



Command line options
--------------------
Additionally to the required command line options discussed above, there are several options to modify the behavior of the |project| experiments.
One set of command line options change the directory structure of the output:

* ``--temp-directory``: Base directory where to write temporary files into (the default is */idiap/temp/$USER/<DATABASE>* when using the grid or */scratch/$USER/<DATABASE>* when executing jobs locally)
* ``--user-directory``: Base directory where to write the results, default is */idiap/user/$USER/<DATABASE>*
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

For that purpose, it is also useful to skip parts of the tool chain.
To do that you can use:

* ``--skip-preprocessing``
* ``--skip-feature-extraction-training``
* ``--skip-feature-extraction``
* ``--skip-projection-training``
* ``--skip-projection``
* ``--skip-enroler-training``
* ``--skip-model-enrolment``
* ``--skip-score-computation``
* ``--skip-concatenation``

although by default files that already exist are not re-created.
To enforce the re-creation of the files, you can use the ``--force`` option, which of course can be combined with the ``--skip...``-options (in which case the skip is preferred).

There are some more command line options that can be specified:

* ``--no-zt-norm``: Disables the computation of the ZT-Norm scores.
* ``--groups``: Enabled to limit the computation to the development ('dev') or test ('eval') group. By default, both groups are evaluated.
* ``--preload-probes``: Speeds up the score computation by loading all probe features (by default, they are loaded each time they are needed). Use this option only, when you are sure that all probe features fit into memory.
* ``--dry-run``: When the grid is enabled, only print the tasks that would have been sent to the grid without actually send them. **WARNING** This command line option is ignored when no ``--grid`` option was specified!


The GBU database
----------------

.. TODO::

  remove this section since this script is outdated.

There is another script *bin/faceverify_gbu.py* that executes experiments on the Good, Bad, and Ugly (GBU) database.
In principle, most of the parameters from above can be used.
One violation is that instead of the ``--models-directories`` option is replaced by only ``--model-directory``.

When running experiments on the GBU database, the default GBU protocol (as provided by `NIST`_) is used.
Hence, training is performed on the special Training set, and experiments are executed using the Target set as models (using a single image for model enrollment) and the Query set as probe.

The GBU protocol does not specify T-Norm-models or Z-Norm-probes, nor it splits off development and test set.
Hence, only a single score file is generated, which might later on be converted into an ROC curve using Bob functions.


The LFW database
----------------
For the `Labeled Faces in the Wild` (LFW) database, there is another script to calculate the experiments, strictly following the LFW protocols.

.. TODO::

  Write the documentation of the LFW script.


Parameter testing
-----------------

.. TODO::

  Write the documentation of the parameter testing script.

.. include:: links.rst
