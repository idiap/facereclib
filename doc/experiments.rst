.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _experiments:

================================
Running non-baseline Experiments
================================

The ``./bin/baselines.py`` script that we have discussed in the previous section is just a wrapper for the ``./bin/faceverify.py`` script.
When the former script is executed, it always prints the call to the latter script.
The latter script is actually doing all the work.
If you want to run experiments with a different setup than the baselines, you should use the ``./bin/faceverify.py`` script directly.

To get the command lines to the ``./bin/faceverify.py`` script that are executed in the baseline experiments, you can call:

.. code-block:: sh

   $ ./bin/baselines.py --all --dry-run

Of course, you can also select a single baseline algorithm to print its command line.

In the following sections the available command line arguments of the ``./bin/faceverify.py`` are listed.
Sometimes, arguments have a long version starting with ``--`` and a short one starting with a single ``-``.
In this section, only the long names of the arguments are listed, please refer to ``./bin/faceverify.py --help`` (or short: ``./bin/faceverify.py -h``) for the abbreviations.

.. _required:

Required Command Line Arguments
-------------------------------
To run a face recognition experiment using the FaceRecLib, you have to tell the ``./bin/faceverify.py`` script, which database, preprocessing, features, and algorithm should be used.
To use this script, you have to specify at least these command line arguments (see also the ``--help`` option):

* ``--database``: The database to run the experiments on, and which protocol to use.
* ``--preprocessing``: The data preprocessing and its parameters.
* ``--features``: The features to extract and their options.
* ``--tool``: The recognition algorithm and all its required parameters.

There is another command line argument that is used to separate the resulting files from different experiments.
Please specify a descriptive name for your experiment to be able to remember, how the experiment was run:

* ``--sub-directory``: A descriptive name for your experiment.


.. _managing-resources:

Managing Resources
~~~~~~~~~~~~~~~~~~
The FaceRecLib is designed in a way that makes it very easy to select the setup of your experiments.
Basically, you can specify your algorithm and its configuration in three different ways:

1. You choose one of the registered resources.
   Just call ``./bin/resources.py`` or ``./bin/faceverify.py --help`` to see, which kind of resources are currently registered.
   Of course, you can also register a new resource.
   How this is done is detailed in section :ref:`register-resources`.

   Example:

   .. code-block:: sh

     $ ./bin/faceverify.py --database atnt

2. You define a configuration file or choose one of the already existing configuration files that are located in `facereclib/configurations`_ and its sub-directories.
   How to define a new configuration file, please read section :ref:`configuration-files`.

   Example:

   .. code-block:: sh

     $ ./bin/faceverify.py --preprocessing facereclib/configurations/preprocessing/tan_triggs.py

3. You directly put the constructor call of the class into the command line.
   Since the parentheses are special characters in the shell, usually you have to enclose the constructor call into quotes.
   If you, e.g., want to extract DCT-block features, just add a to your command line.

   Example:

   .. code-block:: sh

     $ ./bin/faceverify.py --features "facereclib.features.DCTBlocks(block_size=10, block_overlap=0, number_of_dct_coefficients=42)"

   .. note::
     If you use this option with a class that is *not* in the ``facereclib`` module, or your parameters use another module, you have to specify the imports that are needed instead.
     You can do this using the ``--imports`` command line option.
     By default, only the ``facereclib`` is imported.

     Example:

     .. code-block:: sh

       $ ./bin/faceverify.py --tool "facereclib.tools.BIC(comparison_function=numpy.subtract)" --imports facereclib numpy


Of course, you can mix the ways, how you define command line options.

For several databases, preprocessors, feature types, and recognition algorithms the FaceRecLib provides configuration files.
They are located in the `facereclib/configurations`_ directories.
Each configuration file contains the required information for the part of the experiment, and all required parameters are preset with a suitable default value.
Many of these configuration files with their default parameters are registered as resources, so that you don't need to specify the path.

Since the default values might not be optimized or adapted to your problem, you can modify the parameters according to your needs.
The most simple way is to pass the constructor call directly to the command line (i.e., use option 3).
If you want to remember the parameters, you probably would write another configuration file.
In this case, just copy one of the existing configuration files to a directory of your choice, adapt it, and pass the file location to the ``bin/faceverify.py`` script.

In the following, we will provide a detailed explanation of the parameters of the existing :ref:`databases`, :ref:`preprocessors`, :ref:`extractors`, and :ref:`algorithms`.


.. _databases:

Databases
---------
Currently, all implemented databases are taken from Bob_.
To define a common API for all of the databases, the FaceRecLib defines the wrapper classes :py:class:`facereclib.databases.DatabaseBob` and :py:class:`facereclib.databases.DatabaseBobZT` and :py:class:`facereclib.databases.DatabaseFileList` for these databases.
The parameters of this wrapper class are:

Required Parameters
~~~~~~~~~~~~~~~~~~~

* ``name``: The name of the database, in lowercase letters without special characters.
  This name will be used as a default sub-directory to separate resulting files of different experiments.
* ``database = bob.db.<DATABASE>(original_directory=...)``: One of the image databases available at `Idiap at GitHub`_.
  Please set the ``original_directory`` and, if required, the ``original_extension`` parameter in the constructor of that database.
* ``protocol``: The name of the protocol that should be used.
  If omitted, the protocol *Default* will be used (which might not be available in all databases, so please specify).

Optional Parameters
~~~~~~~~~~~~~~~~~~~

These parameters can be used to reduce the number of training images.
Usually, there is no need to specify them, but in case your algorithm requires to much memory:

* ``all_files_option``: The options to the database query that will extract all files.
* ``extractor_training_options``: Special options that are passed to the query, e.g., to reduce the number of images in the extractor training.
* ``projector_training_options``: Special options that are passed to the query, e.g., to reduce the number of images in the projector training.
* ``enroller_training_options``: Special options that are passed to the query, e.g., to reduce the number of images in the enroller training.

Implemented Database Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here we list the database interfaces that are currently available in the FaceRecLib.
By clicking on the database name, you open one configuration file of the database, the link in ``<>`` parentheses will link to the ``bob.db`` database package documentation.
If you have an ``image_directory`` different to the one specified in the file, please change the directory accordingly to be able to use the database.


* :py:class:`facereclib.databases.DatabaseBob`:

  - `AR face <file:../facereclib/configurations/databases/arface.py>`_ <:ref:`bob.db.arface <bob.db.arface>`>: http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html

  .. note::
    At Idiap we might not have the latest version of this database.
    We tried to contact the responsible author of the database, but he didn't reply over years.
    Good luck for your trial to get the data.

  - `AT&T <file:../facereclib/configurations/databases/atnt.py>`_ <:ref:`bob.db.atnt <bob.db.atnt>`>: http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
  - `CAS-PEAL <file:../facereclib/configurations/databases/caspeal.py>`_ <:ref:`bob.db.caspeal <bob.db.caspeal>`>: http://www.jdl.ac.cn/peal/index.html
  - `Face Recognition Grand Challenge ver2.0 (FRGC) <file:../facereclib/configurations/databases/frgc.py>`_ <:ref:`bob.db.frgc <bob.db.frgc>`>: http://www.nist.gov/itl/iad/ig/frgc.cfm

  .. note::
    The FRGC database interface requires to set the ``frgc_directory`` in the configuration file to your copy of the FRGC database.
    If this directory is not set, the FRGC database will not be available, e.g., it will not show up in the available databases of ``./bin/baselines.py --help``.

  - `The Good, The Bad & The Ugly (GBU) <file:../facereclib/configurations/databases/gbu.py>`_ <:ref:`bob.db.gbu <bob.db.gbu>`>: http://www.nist.gov/itl/iad/ig/focs.cfm

  .. note::
    The GBU database uses the data from the MBGC http://www.nist.gov/itl/iad/ig/mbgc.cfm database of the NIST.
    The directory structure of the MBGC seems to be changed lately.
    Hence, the ``bob.db.gbu`` database might not be up to date.
    Please refer to the documentation of this database on how to adapt the database to the new structure.

  - `Labeled Faces in the Wild (LFW) <file:../facereclib/configurations/databases/lfw.py>`_ <:ref:`bob.db.lfw <bob.db.lfw>`>: http://vis-www.cs.umass.edu/lfw/

* :py:class:`facereclib.databases.DatabaseBobZT`:

  - `BANCA <file:../facereclib/configurations/databases/banca.py>`_ <:ref:`bob.db.banca <bob.db.banca>`>: http://www.ee.surrey.ac.uk/CVSSP/banca
  - `MOBIO <file:../facereclib/configurations/databases/mobio.py>`_ <:ref:`bob.db.mobio <bob.db.mobio>`>: http://www.idiap.ch/dataset/mobio
  - `Multi-PIE <file:../facereclib/configurations/databases/multipie.py>`_ <:ref:`bob.db.multipie <bob.db.multipie>`>: http://www.multipie.org
  - `Surveillance Camera (SC) face database <file:../facereclib/configurations/databases/scface.py>`_ <:ref:`bob.db.scface <bob.db.scface>`>: http://www.scface.org
  - `Extended M2VTS (XM2VTS) <file:../facereclib/configurations/databases/xm2vts.py>`_ <:ref:`bob.db.xm2vts <bob.db.xm2vts>`>: http://www.ee.surrey.ac.uk/CVSSP/xm2vtsdb

There is also a special :py:class:`facereclib.databases.DatabaseFileList` interface for the :py:class:`bob.db.verification.filelist.Database` database, which contains a file-based API to define simple evaluation protocols for other databases.
An example, which is based on the `AT&T database`, on how to configure and use this database can be found in `facereclib/tests/databases/atnt_fl/atnt_fl_database.py <file:../facereclib/tests/databases/atnt_fl/atnt_fl_database.py>`_.
For more information, please also read the :ref:`filelist` section.


.. _preprocessors:

Preprocessors
-------------
Currently, all preprocessors that are defined in FaceRecLib perform work on facial images and are, hence, used for face recognition.
Using the :py:class:`bob.ip.base.FaceEyesNorm`, they perform an automatic image alignment to the hand-labeled eye positions as provided by the :ref:`databases`.
Hence, most preprocessors that are defined in `facereclib/preprocessing <file:../facereclib/preprocessing>`_ have a common set of parameters:

Face Cropping Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

* ``cropped_image_size``: The resolution of the cropped image, defined as a tuple (image_height, image_width).
  If not specified, no face cropping is performed.
* ``cropped_positions``: A dictionary of positions where the annotated positions should be placed to.
  For frontal images, usually you define the left and right eye positions: ``{'leye' : (left_eye_y, left_eye_x), 'reye' : (right_eye_y, right_eye_x)}``.
  For profile images, normally you use the mouth and the eye position to crop images: ``{'eye' : (eye_y, eye_x), 'mouth' : (mouth_y, mouth_x)}``.
  This option must be specified when ``cropped_image_size`` is given.
* ``supported_annotations``: A list of pairs of annotation keys that are used to crop the image.
  By default, two kinds of annotations are supported: ``('leye', 'reye')`` and ``('eye', 'mouth')`` (cf. the ``cropped_positions`` option).
  If your database provides different annotations, you can specify them here (also change the annotation keys in the ``cropped_positions`` option).
* ``fixed_positions``: Use these *fixed* positions to crop the face, instead of using the positions provided by the database.
  This option is useful if the database does not provide eye hand-labeled positions, but the images in the database have been aligned before.
  The annotation keys are identical to the ones from the ``cropped_positions`` option.
* ``color_channel``: Use the specified color channel of the colored image.
  Options are ``'gray'``, ``'red'``, ``'green'``, and ``'blue'``, where ``'gray'`` is the default.
  If you want a different color channel, please implement it in the ``gray_channel`` function of the `facereclib/utils/__init__.py <file:../facereclib/utils/__init__.py>`_ file.
* ``offset``: If your feature extraction step needs some surrounding information of the image, you can add an offset here, so that the actual returned image will be larger.

Preprocessor Classes
~~~~~~~~~~~~~~~~~~~~

* :py:class:`facereclib.preprocessing.FaceCrop`: Crops the image to the desired resolution and puts the specified annotations to the specified positions in the image.
* :py:class:`facereclib.preprocessing.HistogramEqualization`: Face cropping and gray value histogram equalization.
* :py:class:`facereclib.preprocessing.TanTriggs`: Face cropping and photometric normalization using the Tan&Triggs algorithm :py:class:`bob.ip.base.TanTriggs`.
  For details about the parameters, please refer to [TT10]_.
* :py:class:`facereclib.preprocessing.SelfQuotientImage`: Face cropping and photometric normalization using the Self-quotient image technology.
  For details about the parameters, please refer to [WLW04]_.
* :py:class:`facereclib.preprocessing.INormLBP`: Face cropping and pixel value normalization using the I-Norm LBP technology.
  For details about the parameters, please refer to [HRM06]_.


.. warning::
  By default, the above listed preprocessors perform image alignment if the database provides eye positions.
  In case there are no eye positions, and no ``fixed_position`` are specified, the image alignment process is silently skipped.
  For larger images, this might lead to problems.

.. note::
  Currently, there is only one database, the `AT&T database`_, that does not provide eye positions since the faces are already cropped.


.. _extractors:

Feature Extractors
------------------
Several different kinds of features can be extracted from the preprocessed data.
Here is the list of classes to perform feature extraction and its parameters.

* :py:class:`facereclib.features.Linearize`: Just puts the elements of the preprocessed data to one vector.
  There are no parameters to this class.

* :py:class:`facereclib.features.Eigenface`: Extracts eigenface features from the preprocessed data, after training the extractor using the preprocessed data from the training set.
  These features are based on the :ref:`bob.learn.linear <bob.learn.linear>` package.

  - ``subspace_dimension``: The number of kept eigenfaces.

* :py:class:`facereclib.features.DCTBlocks`: Extracts *Discrete Cosine Transform* (DCT) features from (overlapping) image blocks.
  These features are based on the :py:class:`bob.ip.base.DCTFeatures` class.
  The default parametrization is the one that performed best on the BANCA database in [WMM+11]_.

  - ``block_size``: The size of the blocks that will be extracted.
    This parameter might be either a single integral value, or a pair ``(blocK_height, block_width)`` of integral values.
  - ``block_overlap``: The overlap of the blocks in vertical and horizontal direction.
    This parameter might be either a single integral value, or a pair ``(blocK_overlap_y, block_overlap_x)`` of integral values.
  - ``number_of_dct_coefficients``: The number of DCT coefficients to use. The actual number will be one less since the first DCT coefficient (which should be 0, if normalization is used) will be removed.
  - ``normalize_blocks``: Normalize the values of the blocks to zero mean and unit standard deviation before extracting DCT coefficients. Default is ``True``.
  - ``normalize_dcts``: Normalize the values of the DCT components to zero mean and unit standard deviation. Default is ``True``.

* :py:class:`facereclib.features.GridGraph`: Extracts Gabor jets in a grid structure [GHW12]_ using functionalities from :ref:`bob.ip.gabor <bob.ip.gabor>`.

  - ``gabor_...``: The parameters of the Gabor wavelet family, with its default values set as given in [WFK97]_.
  - ``normalize_gabor_jets``: Perform Gabor jet normalization during extraction? Default: ``True``.
  - ``extract_gabor_phases``: Store also the Gabor phases, or only the absolute values.
    Possible values are: ``True``, ``False``, ``'inline'``, the default is ``True``.

    .. note::
      Inlining the Gabor phases should not be used when the :py:class:`facereclib.tools.GaborJets` tool is employed.

  - ``eyes``: If given, align the grid to the eye positions.

    + The eye positions can be given as a dictionary ``{'leye' : (left_eye_y, left_eye_x), 'reye' : (right_eye_y, right_eye_x)}``.
      In this case, the parameters ``nodes_between_eyes``, ``nodes_along_eyes``, ``nodes_above_eyes``, and ``nodes_below_eyes`` will be taken into consideration.

    + If ``eyes`` are not specified, a regular grid is placed according to the ``first_node``, ``node_distance``, and ``image_resolution`` parameters.
      In this case, if the ``first_node`` is omitted (i.e. ``None``), it is calculated automatically to equally cover the whole image.

* :py:class:`facereclib.features.LGBPHS`: Extracts *Local Gabor Binary Pattern Histogram Sequences* (LGBPHS) [ZSG+05]_ from the images, using functionality from :ref:`bob.ip.base <bob.ip.base>` and :ref:`bob.ip.gabor <bob.ip.gabor>`:

  - ``block_size``, ``block_overlap``: Setup of the blocks to split the histograms.
    For details on how to use these parameters, please refer to the :py:class:`facereclib.features.DCTBlocks` above.
  - ``gabor_...``: Setup of the Gabor wavelet family.
    The default setup is identical to the :py:class:`facereclib.features.GridGraph`` features.
  - ``use_gabor_phases``: Extract also the Gabor phases (inline) and not only the absolute values -> ELGBPHS [ZSQ+09]_. Default: ``False``.
  - ``lbp_...``: The parameters of the *Local Binary Patterns* (LBP).
    The default values are as given in [ZSG+05]_ (the values of [ZSQ+09]_ might differ).
  - ``sparse_histogram``: Reduces the size of the extracted features, but the computation will take longer. Default: ``False``.
  - ``split_histogram``: Split the extracted histogram sequence.
    Possible values are: ``None``, ``'blocks'``, ``'wavelets'``, ``'both'``, the default is ``None``.

    .. note::
      Splitting the LGBPHS is usually not useful if the employed tool is the :py:class:`facereclib.tools.LGBPHS`.


.. _algorithms:

Recognition Algorithms
----------------------
There are also a variety of recognition algorithms implemented in the FaceRecLib.
All face recognition algorithms are based on the :py:class:`facereclib.tools.Tool` base class.
This base class has parameters that some of the algorithms listed below share.
These parameters mainly deal with how to compute a single score when more than one feature is provided for the model or for the probe:

- ``multiple_model_scoring``: Strategy to combine several features in a model.
  Possible values are (see also :py:func:`facereclib.utils.score_fusion_strategy`):  ``'average'``, ``'min'``, ``'max'``, ``'median'``, default is ``'average'``.
- ``multiple_probe_scoring``: Strategy to combine several probe scores.
  Possible values are (see also :py:func:`facereclib.utils.score_fusion_strategy`):  ``'average'``, ``'min'``, ``'max'``, ``'median'``, default is ``'average'``.

Here is a list of the most important algorithms and their parameters:


* :py:class:`facereclib.tools.PCA`: Computes a PCA projection (:py:class:`bob.learn.linear.PCATrainer`) on the given training features, projects the features to face space and computes the distance of two projected features in face space.

  - ``subspace_dimension``: If integral: the number of kept eigenvalues in the projection matrix; if float in range[0,1]: the percentage of variance to keep.
  - ``distance_function``: The distance function to be used to compare two features in face space. Default: :py:func:`scipy.spatial.distance.euclidean`.
  - ``is_distance_function``: Specifies, if the ``distance_function`` is a distance or a similarity function. Default: ``True``.
  - ``uses_variances``: Does the ``distance_function`` require the PCA variances? Default: ``False``.

* :py:class:`facereclib.tools.LDA`: Computes an LDA (:py:class:`bob.learn.linear.FisherLDATrainer`) or a PCA+LDA projection on the given features.

  - ``lda_subspace_dimension``: **(optional)** Limit the number of dimensions of the LDA subspace.
    If this parameter is not specified, the maximum useful dimensions, i.e, the number of training clients-1 is returned.
    The ``lda_subspace_dimension`` can actually be higher then the useful limit, in which case eigenvectors with vanishing eigenvalues are used.
  - ``pca_subspace_dimension``: **(optional)** If given, the computed projection matrix will be a PCA+LDA matrix, where ``pca_subspace_dimension`` defines the size of the PCA subspace.
    If ``pca_subspace_dimension`` is integral, it is the number of kept eigenvalues in the projection matrix; if is is float, it stands for the percentage of variance to keep.
  - ``distance_function``: The distance function to be used to compare two features in Fisher space. Default: :py:func:`scipy.spatial.distance.euclidean`.
  - ``is_distance_function``: Specifies, if the ``distance_function`` is a distance or a similarity function. Default: ``True``.
  - ``uses_variances``: Does the ``distance_function`` require the LDA variances? Default: ``False``.

    .. note:: If ``lda_subspace_dimension`` is higher than the useful limit, vanishing eigenvalues will be used. In this case, avoid distance functions that require the eigenvalues.

* :py:class:`facereclib.tools.PLDA`: Computes a probabilistic LDA (:py:class:`bob.learn.em.PLDATrainer`)

  - ``subspace_dimension_pca``: **(optional)** If given, features will first be projected into a PCA subspace, and then classified by PLDA.

  .. TODO::
    Document the remaining parameters of the PLDA

* :py:class:`facereclib.tools.BIC`: Computes the Bayesian intrapersonal/extrapersonal classifier (:py:class:`bob.learn.linear.BICTrainer`).
  In this generic implementation, any distance or similarity vector that results as a comparison of two images can be used.
  Currently two different versions are implemented: One with [MWP98]_ and one without (a generalization of [GW09]_) subspace projection of the features.
  A simple configuration file can be found in `facereclib/configurations/tools/bic.py <file:../facereclib/configurations/tools/bic.py>`_, while `facereclib/configurations/tools/bic_jets.py <file:../facereclib/configurations/tools/bic_jets.py>`_ contains a more complex setup including the definition of a particular ``comparison_function``.

  - ``comparison_function``: The function to compare the features in the original feature space.
    For a given pair of features, this function is supposed to compute a vector of similarity (or distance) values.
    In the easiest case, it just computes the element-wise difference of the feature vectors, but more difficult functions can be applied, and the function might be specialized for the features you put in.
  - ``maximum_training_pair_count``: **(optional)** Limit the number of training image pairs to the given value.
  - ``subspace_dimensions``: **(optional)** A tuple of sizes of the intrapersonal and extrapersonal subspaces.
    If given, subspace projection is performed (cf. [MWP98]_) and the subspace projection matrices are truncated to the given sizes.
    If omitted, no subspace projection is performed (cf. [GW09]_).
  - ``uses_dffs``: Use the *Distance From Feature Space* (DFFS) (cf. [MWP98]_) during scoring.
    This flag is only valid when subspace projection is performed, and you should use this flag with care!
  - ``load_function``: A function to load a feature from :py:class:`bob.io.base.HDF5File`. Default: :py:func:`facereclib.utils.load`.
  - ``save_function``: A function to save a feature to :py:class:`bob.io.base.HDF5File`. Default: :py:func:`facereclib.utils.save`.

* :py:class:`facereclib.tools.GaborJets`: Computes a comparison of Gabor jets (:py:class:`bob.ip.gabor.Similarity`).

  - ``gabor_jet_similarity_type``: The Gabor jet similarity to compute.
    Please refer to the documentation of :py:class:`bob.ip.gabor.Similarity` for a list of possible values.
  - ``multiple_feature_scoring``: How to compute the score if several features per model or probe are available.
    Possible values are: ``'average_model'``, ``'average'``, ``'min_jet'``, ``'max_jet'``, ``'med_jet'``, ``'min_graph'``, ``'max_graph'``, ``'med_graph'``, the default is the best working strategy ``'max_jet'``.
  - ``gabor_...``: The parameters of the Gabor wavelet family.
    These parameters are required by some of the Gabor jet similarity functions.
    The default values are identical to the ones in the :py:class:`facereclib.features.GridGraph` features.
    Please assure that this class and the :py:class:`facereclib.features.GridGraph` class get the same configuration, otherwise unexpected things might happen.

* :py:class:`facereclib.tools.LGBPHS`: Computes a similarity measure between extracted LGBP histograms using functions from :ref:`bob.math <bob.math>`.

  - ``distance_function``: The function to be used to compare two histograms. Default: :py:func:`bob.math.chi_square`
  - ``is_distance_function``: Is the given ``distance_function`` a distance (``True``) or a similarity (``False``) function. Default: ``True``

* :py:class:`facereclib.tools.UBMGMM`: Computes a *Gaussian mixture model* (GMM) of the training set (the so-called *Unified Background Model* (UBM) and adapts a client-specific GMM during enrollment (:ref:`bob.learn.em <bob.learn.em>`).

  - ``number_of_gaussians``: The number of Gaussians in the UBM and GMM.
  - ``..._training_iterations``: Maximum number of training iterations of the training steps.

  .. TODO::
    Document the remaining parameters of the UBMGMM tool

* :py:class:`facereclib.tools.ISV`: This class is an extension of the :py:class:`facereclib.tools.UBMGMM`.
  Hence, all the parameters of the :py:class:`facereclib.tools.UBMGMM` must be specified as well.
  Additionally, a subspace projection is computed such that the *Inter Session Variability* of one enrolled client is minimized (:ref:`bob.learn.em <bob.learn.em>`).

  - ``subspace_dimension_of_u``: The dimension of the ISV subspace.

  .. TODO::
    Document the remaining parameters of the ISV tool

* :py:class:`facereclib.tools.JFA`: This class is an extension of the :py:class:`facereclib.tools.UBMGMM`.
  Hence, all the parameters of the :py:class:`facereclib.tools.UBMGMM` must be specified as well.
  Additionally, a subspace projection is computed using the *Joint Factor Analysis* (:ref:`bob.learn.em <bob.learn.em>`).

  - ``subspace_dimension_of_u``: The dimension of the JFA U subspace.
  - ``subspace_dimension_of_v``: The dimension of the JFA V subspace.

  .. TODO::
    Document the JFA tool


Parallel Execution of Experiments
---------------------------------

By default, all jobs of the face recognition tool chain run sequentially on the local machine.
To speed up the processing, some jobs can be parallelized using the SGE_ grid or using multi-processing on the local machine, using the :ref:`GridTK <gridtk>`.
For this purpose, there is another option:

* ``--grid``: The configuration file for the grid execution of the tool chain.

.. note::
  The current SGE setup is specialized for the SGE_ grid at Idiap_.
  If you have an SGE grid outside Idiap_, please contact your administrator to check if the options are valid.

The SGE_ setup is defined in a way that easily allows to parallelize data preprocessing, feature extraction, feature projection, model enrollment, and scoring jobs.
Additionally, if the training of the extractor, projector, or enroller needs special requirements (like more memory), this can be specified as well.

Several configuration files can be found in the `facereclib/configurations/grid <file:../facereclib/configurations/grid>`_ directory.
All of them are based on the :py:class:`facereclib.utils.GridParameters` class.
Here are the parameters that you can set:

* ``grid``: The type of the grid configuration; currently "sge" and "local" are supported.
* ``number_of_preprocessing_jobs``: Number of parallel preprocessing jobs.
* ``number_of_extraction_jobs``: Number of parallel feature extraction jobs.
* ``number_of_projection_jobs``: Number of parallel feature projection jobs.
* ``number_of_enrollment_jobs``: Number of parallel enrollment jobs (when development and evaluation sets are enabled, both sets will be split separately).
* ``number_of_scoring_jobs``: Number of parallel scoring jobs (when development and evaluation sets are enabled, or ZT-norm is computed, more scoring jobs will be generated).

If the ``grid`` parameter is set to ``'sge'`` (the default), jobs will be submitted to the SGE_ grid.
In this case, the SGE_ queue parameters might be specified, either using one of the pre-defined queues (see `facereclib/configurations/grid <file:../facereclib/configurations/grid>`_) or using a dictionary of key/value pairs that are sent to the grid during submission of the jobs:

* ``training_queue``: The queue that is used in any of the training (extractor, projector, enroller) steps.
* ``..._queue``: The queue for the ... step.

If the ``grid`` parameter is set to ``local``, all jobs will be run locally.
In this case, the following parameters for the local submission can be modified:

* ``number_of_parallel_processes``: The number of parallel processes that will be run on the local machine.
* ``scheduler_sleep_time``: The interval in which the local scheduler should check for finished jobs and execute new jobs; the sleep time is given in seconds.

and the ``number_of_..._jobs`` are ignored, and ``number_of_parallel_processes`` is used for all of them.

.. note::
  The parallel execution of jobs on the local machine is currently in BETA status and might be unstable.
  If any problems occur, please file a new bug at http://github.com/idiap/gridtk/issues.

When calling the ``./bin/faceverify.py`` script with the ``--grid ...`` argument, the script will submit all the jobs by taking care of the dependencies between the jobs.
If the jobs are sent to the SGE_ grid (``grid = "sge"``), the script will exit immediately after the job submission.
Otherwise, the jobs will be run locally in parallel and the script will exit after all jobs are finished.

In any of the two cases, the script writes a database file that you can monitor using the ``./bin/jman`` command.
Please refer to ``./bin/jman --help`` or the :ref:`GridTK documentation <gridtk>` to see the command line arguments of this tool.
The name of the database file by default is **submitted.sql3**, but you can change the name (and its path) using the argument:

* ``--submit-db-file``


Command Line Arguments to change Default Behavior
-------------------------------------------------
Additionally to the required command line arguments discussed above, there are several options to modify the behavior of the FaceRecLib experiments.
One set of command line arguments change the directory structure of the output.
By default, the results of the recognition experiment will be written to directory **/idiap/user/<USER>/<DATABASE>/<EXPERIMENT>/<SCOREDIR>/<PROTOCOL>**, while the intermediate (temporary) files are by default written to **/idiap/temp/<USER>/<DATABASE>/<EXPERIMENT>** or **/scratch/<USER>/<DATABASE>/<EXPERIMENT>**, depending on whether the ``--grid`` argument is used or not, respectively:

* <USER>: The Unix username of the person executing the experiments.
* <DATABASE>: The name of the database. It is read from the database configuration.
* <EXPERIMENT>: A user-specified experiment name (see the ``--sub-directory`` argument above).
* <SCOREDIR>: Another user-specified name (``--score-sub-directory`` argument below), e.g., to specify different options of the experiment.
* <PROTOCOL>: The protocol which is read from the database configuration.

These default directories can be overwritten using the following command line arguments, which expects relative or absolute paths:

* ``--temp-directory``
* ``--result-directory`` (for compatibility reasons also ``--user-directory`` can be used)

Re-using Parts of Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to re-use parts previous experiments, you can specify the directories (which are relative to the ``--temp-directory``, but you can also specify absolute paths):

* ``--preprocessed-data-directory``
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
To enforce the re-creation of the files, you can use the:

* ``--force``

argument, which of course can be combined with the ``--skip...`` arguments (in which case the skip is preferred).
To run just a sub-selection of the tool chain, you can also use the:

* ``--execute-only``

argument, which takes a list of options out of: ``preprocessing``, ``extractor-training``, ``extraction``, ``projector-training``, ``projection``, ``enroller-training``, ``enrollment``, ``score-computation``, or ``concatenation``.

Sometimes you just want to try different scoring functions.
In this case, you could simply specify a:

* ``--score-sub-directory``

In this case, no feature or model is recomputed (unless you use the ``--force`` option), but only new scores are computed.

Database-dependent Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Many databases define several protocols that can be executed.
For example, the GBU database provides the three protocols ``'Good'``, ``'Bad'``, ``'Ugly'``.
Usually, the configuration file, e.g., `facereclib.configurations.databases.gbu <File:../facereclib/configurations/databases/gbu.py>`_ contains only one protocol.
To change the protocol, you can either modify the configuration file, or simply use the option:

* ``--protocol``

.. note::
  As an example, to use the ``'Bad'`` protocol of the GBU database you could call:

  .. code-block:: sh

     $ ./bin/faceverify.py --database gbu --protocol Bad ...

Some databases define several kinds of evaluation setups.
For example, often two groups of data are defined, a so-called *development set* and an *evaluation set*.
The scores of the two groups will be concatenated into two files called **scores-dev** and **scores-eval**, which are located in the score directory (see above).
In this case, by default only the development set is employed.
To use both groups, just specify:

* ``--groups dev eval`` (of course, you can also only use the ``'eval'`` set by calling ``--groups eval``)

One score normalization technique is the so-called ZT score normalization.
To enable this, simply use the argument:

* ``--zt-norm``

If the ZT-norm is enabled, two sets of scores will be computed, and they will be placed in two different sub-directories of the score directory, which are by default called **nonorm** and **ztnorm**, but which can be changed using the:

* ``--zt-score-directories``

argument.


Other Arguments
---------------

For some applications it is interesting to get calibrated scores.
Simply add the:

* ``--calibrate-scores``

option and another set of score files will be created by training the score calibration on the scores of the ``'dev'`` group and execute it to all available groups.
The scores will be located at the same directory as the **nonorm** and **ztnorm** scores, and the file names are **calibrated-dev** (and **calibrated-eval** if applicable) .

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

1) Warning messages
2) Informative messages
3) Debug messages

When running experiments locally, my personal preference is verbose level 2, which can be enabled by ``--verbose --verbose``, or using the short version of the argument: ``-vv``.

Finally, there is the:

* ``--dry-run``

argument that can be used for debugging purposes or to check that your command line is proper.
When this argument is used, the experiment is not actually executed, but only the steps that would have been executed are printed to console.

.. note::
  Usually it is a good choice to use the ``--dry-run`` option before submitting jobs to the SGE_, just to make sure that all jobs would be submitted correctly and with the correct dependencies.

.. _facereclib/configurations: file:../facereclib/configurations

.. include:: links.rst
