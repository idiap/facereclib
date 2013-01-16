.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _experiments:

================================
Running non-baseline experiments
================================

The ``bin/baselines.py`` script is just a wrapper for the ``bin/faceverify.py`` script.
When the former script is executed, it always prints the call to the latter script.
The latter script is actually doing all the work.
If you want to run experiments with a different setup than the baselines, you should use the ``bin/faceverify.py`` script directly.

In the following sections the available command line arguments are listed.
Sometimes, arguments have a long version starting with ``--`` and a short one starting with a single ``-``.
In this section, only the long names of the arguments are listed, please refer to ``bin/faceverify.py --help`` (or short: ``bin/faceverify.py -h``) for the abbreviations.

Required command line arguments
-------------------------------
To run a face recognition experiment using the |project|, you have to tell the ``bin/faceverify.py`` script, which database, preprocessing, features, and algorithm should be used.
To use this script, you have to specify at least these command line arguments (see also the ``--help`` option):

* ``--database``: The database and the according protocol.
* ``--preprocessing``: The image preprocessing and its parameters.
* ``--features``: The features to extract and their options.
* ``--tool``: The face recognition algorithm and all its required parameters.

There is another command line argument that is used to separate the resulting files from different experiments.
Please specify a descriptive name for your experiment to be able to remember, how the experiment was run:

* ``--sub-directory``: A descriptive name for your experiment.


.. _managing-resources:

Managing resources
~~~~~~~~~~~~~~~~~~
The |project| is designed in a way that makes it very easy to select
Basically, you can specify your algorithm and its configuration in three different ways:

1. You choose one of the registered resources.
   Just call ``bin/resources.py`` or ``bin/faceverify.py --help`` to see, which kind of resources are currently registered.
   Of course, you can also register a new resource.
   How this is done is detailed in section :ref:`register-resources`.

   Example:
   .. code-block:: sh

     $ bin/faceverify.py --database atnt

2. You define a configuration file or choose one of the already existing configuration files that are located in **facereclib/configurations**.
   How to define a new configuration file, please read section :ref:`configuration-files`.

   Example:
   .. code-block:: sh

     $ bin/faceverify.py --preprocessing facereclib/configurations/preprocessing/tan_triggs.py

3. You directly put the constructor call of the class into the command line.
   If you, e.g., want to extract DCT-block features, just add a to your command line.

   Example:

   .. code-block:: sh

     $ bin/faceverify.py --features "facereclib.features.DCTBlocks(block_size=10, block_overlap=0, number_of_dct_coefficients=42)"

   .. note::
     If you use this option with a class that is *not* in the **facereclib** module, or your parameters use another module, you have to specify the imports that are needed instead.
     You can do this using the ``--imports`` command line option.

     Example:

     .. code-block:: sh

       $ bin/faceverify.py --tool "facereclib.tools.BICTool(distance_function=numpy.subtract)" --imports facereclib numpy


Of course, you can mix the ways, how you define command line options.

For several databases, preprocessors, feature types, and recognition algorithms the |project| provides configuration files.
They are located in the **facereclib/configurations/...** directories.
Each configuration file contains the required information for the part of the experiment, and all required parameters are preset with a suitable default value.
Many of these configuration files with their default parameters are registered as resources, so that you don't need to specify the path.

Since the default values might not be optimized or adapted to your problem, you can modify the parameters according to your needs.
The most simple way is to pass the call directly to the command line.
But if you want to remember the parameters, you probably would write another configuration file.
In this case, just copy one of the existing configuration files to a directory of your choice, adapt it, and pass the file location to the ``bin/faceverify.py`` script.

In the following, we will provide a detailed explanation of the parameters of the existing :ref:`databases`, :ref:`preprocessors`, :ref:`extractors`, and :ref:`algorithms`.


.. _databases:

Databases
~~~~~~~~~
Currently, all implemented databases are taken from Bob_.
To define a common API for all of the databases, the |project| defines the wrapper classes ``facereclib.databases.DatabaseXBob`` and ``facereclib.databases.DatabaseXBobZT`` for these databases.
The parameters of this wrapper class are:

Required parameters:
********************

* ``name``: The name of the database, in lowercase letters without special characters.
  This name will be used as a default sub-directory to separate resulting files of different experiments.
* ``database = xbob.db.<DATABASE>()``: One of the image databases available at `Idiap at GitHub`_.
* ``image_directory``: The base directory, where the image of the database are stored.
* ``image_extension``: The file extension of the images in the database.

Optional parameters:
********************

* ``annotation_directory``: The directory containing the (hand-labeled) annotations of the database, if available.
  If omitted, there will be no image alignment during the preprocessing step.
* ``annotation_extension``: The file extension of the images. Default value: *.pos*.
* ``annotation_type``: The way the annotations are stored in the annotation files.
  Possible values are: *eyecenter*, *named*, *multipie*, *scface*, *cosmin* (see **facereclib/utils/annotations.py** on which one works for you)
  This option must be specified when ``annotation_directory`` is given.
* ``protocol``: The name of the protocol that should be used.
  If omitted, the protocol *Default* will be used (which might not be available in all databases, so please specify).

These parameters can be used to reduce the number of images.
Usually, there is no need to specify them:

* ``all_files_option``: The options to the database query that will extract all files.
* ``extractor_training_options``: Special options that are passed to the query, e.g., to reduce the number of images in the extractor training.
* ``projector_training_options``: Special options that are passed to the query, e.g., to reduce the number of images in the projector training.
* ``enroller_training_options``: Special options that are passed to the query, e.g., to reduce the number of images in the enroller training.

Implemented database interfaces:
********************************

* ``facereclib.database.DatabaseXBob``:

  - AT&T

* ``facereclib.database.DatabaseXBobZT``:

  - BANCA

  .. TODO:
    complete the list of implemented database interfaces and add references to the xbob databases and to the web-sites where to obtain the original data.


.. _preprocessors:

Preprocessors
~~~~~~~~~~~~~
Currently, all image preprocessors that are defined in |project| perform an automatic image alignment to the hand-labeled eye positions as provided by the :ref:`databases`.
Hence, all preprocessors have a common set of parameters:

Face cropping parameters:
*************************

* ``cropped_image_size``: The resolution of the cropped image, defined as a tuple (image_height, image_width).
  If not specified, no face cropping is performed.
* ``cropped_positions``: A dictionary of positions where the annotated positions should be placed to.
  For frontal images, usually you define the left and right eye positions: ``{'leye' : (left_eye_y, left_eye_x), 'reye' : (right_eye_y, right_eye_x)}``.
  For profile images, normally you use the mouth and the eye position to crop images: ``{'eye' : (eye_y, eye_x), 'mouth' : (mouth_y, mouth_x)}``.
  This option must be specified when the cropped_image_size is given.
* ``supported_annotations``: A list of pairs of annotation keys that are used to crop the image.
  By default, two kinds of annotations are supported: ``('leye', 'reye')`` and ``('eye', 'mouth')`` (cf. the ``cropped_positions`` option).
  If your database provides different annotations, you can specify them here (also change the annotation keys in the ``cropped_positions`` option).
* ``fixed_positions``: Use these *fixed* positions to crop the face, instead of using the positions provided by the database.
  This option is useful if the database does not provide eye hand-labeled positions, but the images in the database have been aligned before.
  The annotation keys are identical to the ones from the ``cropped_positions`` option.
* ``color_channel``: Use the specified color channel of the colored image.
  Options are ``'gray'``, ``'red'``, ``'green'``, and ``'blue'``, where ``'gray'`` is the default.
  If you want a different color channel, please implement it in the ``gray_channel`` function of the **facereclib/utils/__init__.py** file.
* ``offset``: If your feature extraction step needs some surrounding information of the image (e.g. the ``LGBPHS`` feature extractor reduces the image size), you can add an offset here, so that the actual returned image will be larger.

Preprocessor classes:
*********************

* ``facereclib.preprocessing.FaceCrop``: Crops the image to the desired resolution and puts the specified annotations to the specified positions in the image.
* ``facereclib.preprocessing.HistogramEqualization``: Face cropping and gray value histogram equalization.
* ``facereclib.preprocessing.TanTriggs``: Face cropping and photometric normalization using the Tan&Triggs algorithm.
  For details about the parameters, please refer to [TT10]_.
* ``facereclib.preprocessing.SelfQuotientImage``: Face cropping and photometric normalization using the Self-quotient image technology.
  For details about the parameters, please refer to [WLW04]_.
* ``facereclib.preprocessing.INormLBP``: Face cropping and pixel value normalization using the I-Norm LBP technology
  For details about the parameters, please refer to [HRM06]_.



.. _extractors:

Feature extractors
~~~~~~~~~~~~~~~~~~
Several different kinds of features can be extracted from the preprocessed images.
Here is the list of classes to perform feature extraction and its parameters.

* ``facereclib.features.Linearize``: Just extracts the pixels of the image to one vector.
  There are no parameters to this class.

* ``facereclib.features.Eigenface``: Extract eigenface features from the images.

  - ``subspace_dimension``: The number of kept eigenfaces.

* ``facereclib.features.DCTBlocks``: Extracts *Discrete Cosine Transform* (DCT) features from (overlapping) image blocks.
  The default parametrization is the one that performed best on the BANCA database in [WMM+11]_.

  - ``block_size``: The size of the blocks that will be extracted.
    This parameter might be either a single integral value, or a pair ``(blocK_height, block_width)`` of integral values.
  - ``block_overlap``: The overlap of the blocks in vertical and horizontal direction.
    This parameter might be either a single integral value, or a pair ``(blocK_overlap_y, block_overlap_x)`` of integral values.
  - ``number_of_dct_coefficients``: The number of DCT coefficients to use.

* ``facereclib.features.GridGraph``: Extracts Gabor jets in a grid structure [GHW12]_.

  - ``gabor_...``: The parameters of the Gabor wavelet family, with its default values set as given in [WFK97]_.
  - ``normalize_gabor_jets``: Perform Gabor jet normalization during extraction?
  - ``extract_gabor_phases``: Store also the Gabor phases, or only the absolute values.
    Possible values are: ``True``, ``False``, ``'inline'``, the default is ``True``.

    .. note::
      Inlining the Gabor phases should not be used when the ``facereclib.tools.GaborJetTool`` tool is employed.

  - ``eyes``: If given, align the grid to the eye positions.
    The eye positions must be given as a dictionary ``{'leye' : (left_eye_y, left_eye_x), 'reye' : (right_eye_y, right_eye_x)}``.
    In this case, the parameters ``nodes_between_eyes``, ``nodes_along_eyes``, ``nodes_above_eyes``, and ``nodes_below_eyes`` will be taken into consideration.
    If ``eyes`` are not specified, a regular grid is placed according to the ``first_node``, ``node_distance``, and ``image_resolution`` parameters.
    In this case, if the ``first_node```is not given (i.e. ``None``), it is calculated automatically to equally cover the whole image.

* ``facereclib.features.LGBPHS``: Extracts *Local Gabor Binary Pattern Histogram Sequences* (LGBPHS) [ZSG+05]_ from the images:

  - ``block_size``, ``block_overlap``: Setup of the blocks to split the histograms.
    For details on how to use these parameters, please refer to the ``facereclib.features.DCTBlocks`` above.
  - ``gabor_...``: Setup of the Gabor wavelet family.
    The default setup is identical to the ``facereclib.features.GridGraph`` features.
  - ``use_gabor_phases``: Extract also the Gabor phases (inline) and not only the absolute values [ZSQ+09]_.
  - ``lbp_...``: The parameters of the *Local Binary Patterns* (LBP).
    The default values are as given in [ZSG+05]_ (the values of [ZSQ+09]_ might differ).
  - ``sparse_histogram``: Reduces the size of the extracted features, but the computation will take longer.
  - ``split_histogram``: Split the extracted histogram sequence.
    Possible values are: ``None``, ``'blocks'``, ``'wavelets'``, ``'both'``

    .. note::
      Splitting the LGBPHS is usually not useful if the employed tool is the ``LGBPHSTool``.


.. _algorithms:

Face recognition algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are also a variety of face recognition algorithms implemented in the |project|.
Here is a list of the most important algorithms and their parameters:


* ``facereclib.tools.PCATool``: Computes a PCA projection on the given features and computes the distance of two projected features in eigenspace.

  - ``subspace_dimension``: If integral: the number of kept eigenvalues in the projection matrix; if float: the percentage of variance to keep.
  - ``distance_function``: The distance function to be used to compare two features in eigenspace.
  - ``is_distance_function``: Specifies, if the ``distance_function`` is a distance or a similarity function.
  - ``multiple_feature_scoring``: How to compute the score if several features per model are available, possible values: ``average``, ``min``, ``max``, ``median``

* ``facereclib.tools.LDATool``: Computes an LDA or a PCA+LDA projection on the given features.

  - ``lda_subspace_dimension``: **(optional)** Limit the number of dimensions of the LDA subspace.
    If this parameter is not specified, no truncation is applied.
  - ``pca_subspace_dimension``: **(optional)** If given, the computed projection matrix will be a PCA+LDA matrix, where ``pca_subspace_dimension`` defines the size of the PCA subspace.
    If ``pca_subspace_dimension`` is integral, it is the number of kept eigenvalues in the projection matrix; if is is float, it stands for the percentage of variance to keep.
  - ``distance_function``: The distance function to be used to compare two features in Fisher space.
  - ``is_distance_function``: Specifies, if the ``distance_function`` is a distance or a similarity function.
  - ``multiple_feature_scoring``: How to compute the score if several features per model are available, possible values: ``average``, ``min``, ``max``, ``median``

* ``facereclib.tools.PLDATool``: Computes a probabilistic LDA

  - ``subspace_dimension_pca``: **(optional)** If given, features will first be projected into a PCA subspace, and then classified by PLDA.

  .. TODO::
    Document the remaining parameters of the PLDATool

* ``facereclib.tools.BICTool``: Computes the Bayesian intrapersonal/extrapersonal classifier.
  Currently two different versions are implemented: One with [MWP98]_ and one without (a generalization of [GW09]_) subspace projection of the features.

  - ``distance_function``: The function to compare the features in the original feature space.
    For a given pair of features, this function is supposed to compute a vector of similarity (or distance) values.
    In the easiest case, it just computes the element-wise difference of the feature vectors, but more difficult functions can be applied, and the function might be specialized for the features you put in.
  - ``maximum_training_pair_count``: **(optional)** Limit the number of training image pairs to the given value.
  - ``subspace_dimensions``: **(optional)** A tuple of sizes of the intrapersonal and extrapersonal subspaces.
    If given, subspace projection is performed (cf. [MWP98]_) and the subspace projection matrices are truncated to the given sizes.
    If omitted, no subspace projection is performed (cf. [GW09]_).
  - ``uses_dffs``: Use the *Distance From Feature Space* (DFFS) (cf. [MWP98]_) during scoring.
    This flag is only valid when subspace projection is performed, and you should use this flag with care!

* ``facereclib.tools.GaborJetTool``: Computes a comparison of Gabor jets stored in grid graphs.

  - ``gabor_jet_similarity_type``: The Gabor jet similarity to compute.
    Please refer to the documentation of Bob_ for a list of possible values.
  - ``multiple_feature_scoring``: How to compute the score if several features per model are available.
    Possible values are: 'average_model', 'average', 'min_jet', 'max_jet', 'med_jet', 'min_graph', 'max_graph', 'med_graph'.
  - ``gabor_...``: The parameters of the Gabor wavelet family.
    These parameters are required by some of the Gabor jet similarity functions.
    The default values are identical to the ones in the ``facereclib.features.GridGraph`` features.

* ``facereclib.tools.LGBPHSTool``: Computes a similarity measure between extracted LGBP histograms.

  - ``distance_function``: The function to be used to compare two histograms.
  - ``is_distance_function``: Is the given ``distance_function`` a distance (``True``) or a similarity (``False``) function.

* ``facereclib.tools.UBMGMMTool``: Computes a *Gaussian mixture model* (GMM) of the training set (the so-called *Unified Background Model* (UBM) and adapts a client-specific GMM during enrollment.

  - ``number_of_gaussians``: The number of Gaussians in the UBM and GMM.
  - ``..._training_iterations``: Maximum number of training iterations of the training steps.

  .. TODO::
    Document the remaining parameters of the UBMGMMTool

* ``facereclib.tools.ISVTool``: This class is an extension of the ``facereclib.tools.UBMGMMTool``.
  Hence, all the parameters of the ``facereclib.tools.UBMGMMTool`` must be specified as well.
  Additionally, a subspace projection is computed such that the *Inter Session Variability* of one enrolled client is minimized.

  - ``subspace_dimension_of_u``: The dimension of the ISV subspace.

  .. TODO::
    Document the remaining parameters of the ISVTool

.. TODO::
  Document the JFATool


Parameters of the SGE grid:
---------------------------
By default, all jobs of the face recognition tool chain run sequentially on the local machine.
To speed up the processing, some jobs can be parallelized using the SGE grid.
For this purpose, there is another option

* ``--grid``: The configuration file for the grid execution of the tool chain.

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

In this case, no feature or model is recomputed (unless you use the ``--force`` option), but only the scores are re-calculated.

Database-dependent arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Some databases define several sets of testing setups.
For example, often two groups of images are defined, a so-called *development set* and an *evaluation set*.
The scores of the two groups will be concatenated into two files called **scores-dev** and **scores-eval**, which are located in the score directory (see above).
In this case, by default only the development set is employed.
To use both groups, just specify:

* ``--groups dev eval`` (of course, you can also only use the 'eval' set by calling ``--groups eval``)

One score normalization technique is the so-called ZT score normalization.
To enable this, simply use the argument:

* ``--zt-norm``

If the normalization is enabled, two sets of scores will be computed, and they will be placed in two different sub-directories of the score directory, which are by default called **nonorm** and **ztnorm**, but which can be changed using the:

* ``--zt-score-directories``

argument.


Other arguments
---------------
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

argument that can be used for debugging purposes or to check that your command line is proper.
When this argument is used, the experiment is not actually executed, but only the steps that


.. include:: links.rst
