.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

Running specialized experiments
===============================

The ``bin/baselines.py`` script is just a wrapper for the ``bin/faceverify.py`` script.
When the former script is executed, it always prints the call to the latter script.
The latter script is actually doing all the work.
If you want to run experiments with a different setup than the baselines, you should use the ``bin/faceverify.py`` script directly.

In the following sections the available command line arguments are listed.
Sometimes, arguments have a long version starting with ``--`` and a short one starting with a single `-`.
In this section, only the long names of the arguments are listed, please refer to ``bin/faceverify.py --help`` (or short: ``bin/faceverify.py -h``) for the abbreviations.

Required command line arguments
-------------------------------
To use this script, you have to specify at least these command line arguments (see also the ``--help`` option):

* ``--database``: The configuration file for the database.
* ``--preprocessing``: The configuration file for image preprocessing.
* ``--features``: The configuration file for feature extraction.
* ``--tool``: The configuration file for the face verification algorithm.
* ``--sub-directory``: A descriptive name for your experiment.

For several databases, preprocessings, feature types, and recognition algorithms the |project| provides configuration files.
They are located in the **config/...** directories.
Each configuration file contains the required information for the part of the experiment.
All required parameters are pre-set with a suitable default value, but they might not be optimized or adapted to your problem.
Hence, you can modify the parameters according to your needs.
You can also create new configuration files.

Here is a list of parameters that the configuration file requires:

Database
~~~~~~~~
Required parameters:

* ``name``: The name of the database, in lowercase letters.
  In principle, this name should be identical to the name of the file.
* ``db = xbob.db.<DATABASE>``: One of the image databases available at `Idiap at GitHub`_.
* ``protocol``: The name of the protocol that should be used.
* ``image_directory``: The base directory, where the image of the database are stored.
* ``image_extension``: The file extension of the images in the database.

Optional parameters:

* ``annotation_directory``: The directory containing the (hand-labeled) annotations of the database; if omitted, there will be no image alignment during the preprocessing step.
* ``annotation_extension``: The file extension of the images; default: '.pos'
* ``annotation_type``: The way the annotations are stored in the annotation files; possible values: 'eyecenter', 'named', 'multipie', 'scface', 'cosmin' (see **facereclib/utils/annotations.py** on which one works for you); must be specified when ``annotation_directory`` is given.
* ``all_files_option``: The options to the database query that will extract all files.

These parameters can be used to reduce the number of training images.
Usually, there is no need to specify them:

* ``extractor_training_options``: Special options that are passed to the query, e.g., to reduce the number of images in the extractor training.
* ``projector_training_options``: Special options that are passed to the query, e.g., to reduce the number of images in the projector training.
* ``enroller_training_options``: Special options that are passed to the query, e.g., to reduce the number of images in the enroller training.
* ``features_by_clients_options``: Special options that are passed to the query to retrieve features only for some clients.

Preprocessing
~~~~~~~~~~~~~
Required parameters:

* ``preprocessor = facereclib.preprocessing.<PREPROCESSOR>``: The preprocessor that should be used.

Parameters used for image alignment:

* ``CROPPED_IMAGE_HEIGHT``, ``CROPPED_IMAGE_WIDTH``: The size of the cropped image.
* ``RIGHT_EYE_POS``, ``LEFT_EYE_POS``: The positions in the cropped image, at which the eyes should be placed, if the image is frontal.
  Note that usually the ``LEFT_EYE_POS`` is to the right of the ``RIGHT_EYE_POS``.
* ``EYE_POS``, ``MOUTH_POS``: The positions in the cropped image, at which the eye and the mouth should be placed, if the image is a profile image.
  Note that you usually should not use the same parameters for left and right profile images.
* ``OFFSET``: If your latter feature extraction stage needs an offset for extracting the information (e.g., the LGBPHS features), you can specify it here.
* ``FIXED_RIGHT_EYE``, ``FIXED_LEFT_EYE``: **(optional)** These positions are taken to be the hand-labeled eye positions in the original images, e.g., if the database does not provide annotations.

Other common parameters for all preprocessors:

* ``COLOR_CHANNEL``: The color channel of the image to be used; possible values: 'gray', 'red', 'green', 'blue'

Parameters of the implemented <PREPROCESSOR>'s:

* ``FaceCrop``: No further parameter required.
* ``HistogramEqualization``: No further parameter required.
* ``TanTriggs``: The parameters of the Tan&Triggs algorithm; for details please refer to [TT10]_.
* ``SelfQuotientImage``: The variance of the self quotient image; for details please refer to [WLW04]_.
* ``INormLBP``: The parametrization of the LBP extractor; for details please refer to [HRM06]_.

.. [TT10]   X. Tan and B. Triggs. Enhanced local texture feature sets for face recognition under difficult lighting conditions. IEEE Transactions on Image Processing, 19(6):1635-1650, 2010.
.. [WLW04]  H. Wang, S.Z. Li and Y. Wang. Face recognition under varying lighting conditions using self quotient image. Proceedings of the Sixth IEEE International Conference on Automatic Face and Gesture Recognition, pages 819-824. 2004.
.. [HRM06]  G. Heusch, Y. Rodriguez, and S. Marcel. Local Binary Patterns as an Image Preprocessing for Face Authentication. In IEEE International Conference on Automatic Face and Gesture Recognition (AFGR), 2006.

Feature extraction:
~~~~~~~~~~~~~~~~~~~
Required parameters:

* ``feature_extractor = facereclib.features.<FEATURE_EXTRACTOR>``: The class that should be used for feature extraction.

Implemented <FEATURE_EXTRACTOR>'s:

* ``Linearize``: Just extracts the pixels of the image to one vector; no parameters.

* ``Eigenface``: Extract eigenface features from the images:

  - ``SUBSPACE_DIMENSION``: The number of kept eigenfaces.

* ``DCTBlocks``: Extracts *Discrete Cosine Transform* (DCT) features from (overlapping) image blocks:

  - ``BLOCK_HEIGHT``, ``BLOCK_WIDTH``: The size of the blocks that will be extracted.
  - ``BLOCK_Y_OVERLAP``, ``BLOCK_X_OVERLAP``: The overlap of the blocks in vertical and horizontal direction.
  - ``NUMBER_OF_DCT_COEFFICIENTS``: The number of DCT coefficients to keep.

* ``GridGraph``: Extracts Gabor jets in a grid structure:

  - ``GABOR_...``: The parameters of the Gabor wavelet family.
  - ``NORMALIZE_GABOR_JETS``: Perform Gabor jet normalization during extraction?
  - ``EXTRACT_GABOR_PHASES``: Store also the Gabor phases, or only the absolute values; possible values: True, False, 'inline'.
  - **(a)** ``FIRST_NODE``, ``LAST_NODE``, ``NODE_DISTANCE``: Place the nodes in a regular grid according to the given parameters.
  - **(b)** ``NODE_COUNT_..._EYES``: Place a regular grid such that two nodes are placed at the eye positions (and distribute the remaining nodes accordingly).

.. note::
  Inlining the Gabor phases should not be used when the ``GaborJetTool`` tool is employed.

* ``LGBPHS``: Extracts *Local Gabor Binary Pattern Histogram Sequences* from the images:

  - ``BLOCK_...``: Setup of the blocks to split the histograms, see above.
  - ``GABOR_...``: Setup of the Gabor wavelet family.
  - ``USE_SPARSE_HISTOGRAM``: Reduce the size of the extracted features; the computation will take longer.
  - ``SPLIT_HISTOGRAM``: Split the extracted histogram sequence; possible values: None, 'blocks', 'wavelets', 'both'; might be useful if the employed tool is not ``LGBPHSTool``.
  - ``USE_GABOR_PHASES``: Extract also the Gabor phases (inline) and not only the absolute values.
  - Remaining: Parameters of the *Local Binary Patterns* (LBP)

Face recognition algorithms:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Required parameters:

* `tool = facereclib.tools.<TOOL>``: The face recognition algorithm.

Implemented <TOOL>'s:

* ``PCATool``: Computes a PCA projection on the given features.

  - ``SUBSPACE_DIMENSION``: The number of kept eigenvalues.
  - ``distance_function``: The distance function to be used to compare two features in eigenspace.

* ``LDATool``: Computes an LDA or a PCA+LDA projection on the given features.

  - ``LDA_SUBSPACE_DIMENSION``: **(optional)** Limit the number of dimensions of the LDA subspace; if not specified, no truncation is applied.
  - ``PCA_SUBSPACE_DIMENSION``: **(optional)** If given, the computed projection matrix will be a PCA+LDA matrix, where ``PCA_SUBSPACE_DIMENSION`` defines the size of the PCA subspace.

* ``PLDATool``: Computes a probabilistic LDA

  - ``SUBSPACE_DIMENSION_PCA``: **(optional)** If given, features will first be projected into a PCA subspace, and then classified by PLDA

.. TODO::
  Document the remaining parameters of the PLDATool

* ``BICTool``: Computes the Bayesian intrapersonal/extrapersonal classifier.
  Currently two different versions are implemented: One with [] and one without [] subspace projection of the features.

  - ``MAXIMUM_TRAINING_PAIR_COUNT``: **(optional)** Limit the number of training image pairs to the given value.
  - ``INTRA_SUBSPACE_DIMENSION``, ``EXTRA_SUBSPACE_DIMENSION``: **(optional)** The size of the intrapersonal and extrapersonal subspaces; if omitted, no subspace projection is performed.
  - ``USE_DFFS``: Use the *Distance From Feature Space* (DFFS) during scoring; only valid when subspace projection is performed; use this flag with care!
  - ``distance_function``: The function to compare the features in the original feature space.
    For a given pair of features, this function is supposed to compute a vector of similarity (or distance) values.
    In the easiest case, it just computes the element-wise difference of the feature vectors, but more difficult functions can be applied, and the function might be specialized for the features you put in.

* ``GaborJetTool``: Computes a comparison of Gabor jets stored in grid graphs.

  - ``EXTRACT_AVERAGED_MODELS``: Enroll models by averaging the graphs using linear interpolation of the Gabor jets, or simply store all Gabor graphs.
  - ``GABOR_...``: The parameters of the Gabor wavelet family; required by some of the Gabor jet similarity functions.
  - ``GABOR_JET_SIMILARITY_TYPE``: The Gabor jet similarity to compute.

* ``LGBPHSTool``: Computes a similarity measure between extracted LGBP histograms.

  - ``distance_function``: The function to be used to compare two histograms.
  - ``IS_DISTANCE_FUNCTION``: Is the given ``distance_function`` a distance or a similarity function.

* ``UBMGMMTool``: Computes a *Gaussian mixture model* (GMM) of the training set (the so-called *Unified Background Model* (UBM) and adapts a client-specific GMM during enrollment.

  - ``GAUSSIANS``: The number of Gaussians in the UBM and GMM
  - ``..._TRAINING_ITERATIONS``: Maximum number of training iterations of the training steps

.. TODO::
  Document the remaining parameters of the UBMGMMTool

* ``ISVTool``: Adaption of the UBMGMMTool; additionally, a subspace projection is computed such that the *Inter Session Variability* of one enrolled client is minimized.

  - ``GAUSSIANS``: The number of Gaussians in the UBM and GMM
  - ``..._TRAINING_ITERATIONS``: Maximum number of training iterations of the training steps
  - ``SUBSPACE_DIMENSION_OF_U``: The dimension of the ISV subspace

.. TODO::
  Document the remaining parameters of the UBMGMMTool

.. TODO::
  Document the JFATool


Parameters of the SGE grid:
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When no ``--grid`` argument is specified, all jobs will run sequentially on the local machine.
To speed up the processing, some jobs can be parallelized using the SGE grid.

.. note::
  The current SGE setup is specialized for the SGE grid at Idiap_.
  If you have an SGE grid outside Idiap_, please contact your administrator to check if the options are valid.

The SGE setup is defined in a way that easily allows to parallelize image preprocessing, feature extraction, feature projection, model enrollment, and scoring jobs.
Additionally, if the training of the extractor, projector, or enroller needs special requirements (like more memory), this can be specified as well.
Here are the parameters that you can set:

* ``training_queue``: Parametrization of the queue that is used in any of the training (extractor, projector, enroller)
* ``..._queue``: The queue for the ... step
* ``number_of_images_per_job``: Number of images that one preprocessing job should handle.
* ``number_of_features_per_job``: Number of images that one feature extraction job should handle.
* ``number_of_projections_per_job``: Number of features that one feature projection job should handle.
* ``number_of_models_per_enroll_job``: Number of models that one enroll job should enroll.
* ``number_of_models_per_score_job``: Number of models for which on score job should compute the scores.

When calling the ``bin/faceverify.py`` script with the ``--grid ...`` argument, the script will submit all the jobs to the SGE grid and executes.
It will write a database file that you can monitor using the ``bin/jman`` command.
Please refer to ``bin/jman --help`` to see the command line arguments of this tool.
The name of the database file by default is **submitted.db**, but you can change the name (and its path) using the argument:

* ``--submit-db-file``


Command line arguments to change default behavior
-------------------------------------------------

Additionally to the required command line arguments discussed above, there are several options to modify the behavior of the |project| experiments.
One set of command line arguments change the directory structure of the output.
By default, the verification result will be written to directory **/idiap/user/<USER>/<DATABASE>/<EXPERIMENT>/<SCOREDIR>/<PROTOCOL>**, while the intermediate (temporary) files are by default written to **/idiap/temp/<USER>/<DATABASE>/<EXPERIMENT>** or **/scratch/<USER>/<DATABASE>/<EXPERIMENT>**, depending on whether the ``--grid`` argument is used or not, respectively:

* <USER>: The Unix username of the person executing the experiments.
* <DATABASE>: The name of the database. It is read from the database configuration file.
* <EXPERIMENT>: A user-specified experiment name (see the ``--sub-directory`` argument above).
* <SCOREDIR>: Another user-specified name (``--score-sub-directory`` argument below), e.g., to specify different options of the experiment.
* <PROTOCOL>: The protocol which is read from the database configuration file.

Changing directories
~~~~~~~~~~~~~~~~~~~~

These default directories can be overwritten using the following command line arguments, which expects relative or absolute paths:

* ``--temp-directory``
* ``--user-directory``

Re-using parts of experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to re-use parts previous experiments, you can specify the directories (which are relative to the ``--temp-directory``, but you can also specify absolute paths):

* ``--preprocessed-image-directory``
* ``--features-directory``
* ``--projected-directory``
* ``--models-directories`` (one for each the models and the ZT-norm-models, see below)

or even trained extractor, projector, or enroller (i.e., the results of the extractor, projector, or enroller training):

* ``--extractor-file``
* ``--projector-file``
* ``--enroller-file``

For that purpose, it is also useful to skip parts of the tool chain.
To do that you can use:

* ``--skip-preprocessing``
* ``--skip-extractor-training``
* ``--skip-extraction``
* ``--skip-projector-training``
* ``--skip-projection``
* ``--skip-enroller-training``
* ``--skip-enrollment``
* ``--skip-score-computation``
* ``--skip-concatenation``

although by default files that already exist are not re-created.
To enforce the re-creation of the files, you can use the

* ``--force``

argument, which of course can be combined with the ``--skip...`` arguments (in which case the skip is preferred).

Sometimes you just want to try different scoring functions.
In this case, you could simply specify a:

* ``--score-sub-directory``

In this case, no feature or model is recomputed, but only the scores are re-calculated.

Database-dependent arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some databases define several sets of testing setups.
For example, often two groups of images are defined, a so-called *development set* and an *evaluation set*.
The scores of the two groups will be concatenated into two files called **scores-dev** and **scores-eval**, which are located in the score directory (see above).
In this case, by default only the development set is employed.
To use both groups, just specify:

* ``--groups dev eval`` (of course, you can also only use the 'eval' set by calling ``--groups eval``)

One score normalization technique is the so-called ZT normalization.
To enable this, simply use the argument:

* ``--zt-norm``

If the normalization is enabled, two sets of scores will be computed, and they will be placed in two different sub-directories of the score directory, which are by default called **nonorm** and **ztnorm**, but which can be changed using the:

* ``--zt-score-directories``

argument.


Other arguments
~~~~~~~~~~~~~~~

During score computation, the probe files usually will be loaded on need.
Since file IO might take a while, you might want to use the argument:

* ``--preload-probes``

that loads all probe files into memory.

.. warning::
  Use this argument with care.
  For some feature types and/or image databases, the memory required by the features is huge.

By default, the algorithms are set up to execute quietly, and only errors are reported.
To change this behavior, you can -- again -- use the

* ``--verbose``

argument several times to increase the verbosity level to show:

1: Warning messages
2: Informative messages
3: Debug messages

When running experiments locally, my personal preference is verbose level 2, which can be enabled by ``--verbose --verbose``, or using the short version of the argument: ``-vv``.

Finally, there is the:

* ``--dry-run``

argument that can be used for debugging purposes.
When this argument is used, you only


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
