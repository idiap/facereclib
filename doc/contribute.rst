.. vim: set fileencoding=utf-8 :
.. Manuel Guenther <Manuel.Guenther@idiap.ch>
.. Mon 23 04 2012

=========================================================================================
Implementing your own Database, Preprocessor, Feature Extractor, or Recognition Algorithm
=========================================================================================

The FaceRecLib module is specifically designed to be as flexible as possible while trying to keep things simple.
Therefore, it uses python to implement algorithms.
It is file based so any algorithm can implement its own way of reading and writing data, features or models.
Algorithm configurations are stored in configuration files, so it should be easy to test different parameters of your algorithms without modifying the code.

To implement your own database, preprocessor, feature, or algorithm, simply follow the examples that are already in the FaceRecLib.
In the following sections there will be an overview of the functions that need to be implemented.

The FaceRecLib is designed in a way that useful default functionalities are executed.
If you want/need to have a different behavior, you can simply add functions to your classes and register these functions, for details please see below.


Implementing your own Functions
-------------------------------

There are two options to add functionality to the FaceRecLib.
The preferred option should be to write a satellite package of the FaceRecLib, implement everything you want to do, test it and document it.
Please read the :ref:`satellite-packages` section for more details on this.

Here, we describe the second way, which is to add functionality to FaceRecLib directly.

Base Classes
~~~~~~~~~~~~
In general, any database, preprocessor, feature extractor or recognition algorithm should be derived from a base class that is detailed below.
This base class provides default implementations of functionality that can be used directly or overwritten in your class.
One of these functions, which is identical to all base classes, is the ``__str__(self)`` function, a special Python construct to convert an object of a class into a string that contains information about the object.
In the FaceRecLib, this function is used to write the experimental configuration into a specific text file (by default: **Experiment.info** in the ``--result-directory``).
This information is useful to see the exact configuration of the experiment with which the results was generated.

There are two ways of providing these information for your class:

1. Call the base class constructor and specify all parameters that should be added to the information file.
2. Overwrite the ``__str__(self)`` function in your class, following the example of the base class.

.. _filelist:

Image Databases
~~~~~~~~~~~~~~~
If you have your own database that you want to execute the recognition experiments on, you should first check if you could use the :ref:`Verifcation FileList Database <bob.db.verification.filelist>` interface by defining appropriate file lists for the training set, the model set, and the probes.
If you can do this, just write your own configuration file that uses the :py:class:`facereclib.databases.DatabaseFileList` interface (see :ref:`databases` for details).
In most of the cases, the :py:class:`bob.db.verification.filelist.Database` should be sufficient to run experiments.
Please refer to the documentation :ref:`Documentation <bob.db.verification.filelist>` of this database for more instructions on how to configure this database.

In case you want to have a more complicated interface to your database, you are welcome to write your own database wrapper class.
In this case, you have to derive your class from the :py:class:`facereclib.databases.Database`, and provide the following functions:

* ``__init__(self, <your-parameters>, **kwargs)``: Constructor of your database interface.
  Please call the base class constructor, providing all the required parameters (see :ref:`databases`), e.g. by ``facereclib.databases.Database.__init__(self, **kwargs)``.
* ``all_files(self)``: Returns a list of all :py:class:`facereclib.databases.File` objects of the database.
  The list needs to be sorted by the file id (you can use the ``self.sort(files)`` function for sorting).
* ``training_files(self, step, arrange_by_client = False)``: A sorted list of the :py:class:`facereclib.databases.File` objects that is used for training.
  If ``arrange_by_clients`` is enabled, you might want to use the ``self.arrange_by_client(files)`` function to perform the job.
* ``model_ids(self, group = 'dev'): The ids for the models (usually, there is only one model per client and, thus, you can simply use the client ids) for the given group.
  Usually, providing ids for the group ``'dev'`` should be sufficient.
* ``client_id_from_model_id(self, model_id)``: Returns the client id for the given model id.
* ``enroll_files(self, model_id, group='dev')``: Returns the list of model :py:class:`facereclib.databases.File` objects for the given model id.
* ``probe_files(self, model_id=None, group='dev')``: Returns the list of probe files, the given model_id should be compared with.
  Usually, all probe files are compared with all model files.
  In this case, you can just ignore the ``model_id``.
  If the ``model_id`` is ``None``, this function is supposed to return *all* probe files for all models of the given group.

Additionally, you can define more lists that can be used for ZT score normalization.
In this case, derive you class from :py:class:`facereclib.databases.DatabaseZT` instead, and additionally overwrite the following functions:

* ``t_model_ids(self, group = 'dev')``: The ids for the T-Norm models for the given group.
* ``t_enroll_files(self, model_id, group='dev')``: Returns the list of model :py:class:`facereclib.databases.File` objects for the given T-Norm model id.
* ``z_probe_files(self, group='dev')``: Returns the list of Z-probe :py:class:`facereclib.databases.File` objects, with which all the models and T-Norm models are compared.

.. note:
  For a proper face recognition protocol, the identities from the models and the T-Norm models, as well as the Z-probes should be different.

For some protocols, a single probe consists of several features, see :ref:`algorithms` about strategies how to incorporate several probe files into one score.
If your database should provide this functionality, please overwrite:

* ``uses_probe_file_sets(self)``: Return ``True`` if the current protocol of the database provides multiple files for one probe.
* ``probe_file_sets(self, model_id=None, group='dev')``: Returns a list of lists of :py:class:`facereclib.databases.FileSet` objects.
* ``z_probe_file_sets(self, model_id=None, group='dev')``: Returns a list of lists of Z-probe :py:class:`facereclib.databases.FileSet` objects (only needed if the base class is :py:class:`facereclib.databases.DatabaseZT`).


Data Preprocessors
~~~~~~~~~~~~~~~~~~
All preprocessing classes should be derived from the :py:class:`facereclib.preprocessing.Preprocessor` class.
In your class, please overload the functions:

* ``__init__(self, <parameters>)``: Initializes the image preprocessing algorithm with the parameters it needs.
  Please call the base class constructor in this constructor, e.g. as ``facereclib.preprocessing.Preprocessor.__init__(self)``.
* ``__call__(self, original_data, annotations) -> data``: preprocesses the data given the dictionary of annotations (e.g. ``{'reye' : [re_y, re_x], 'leye': [le_y, le_x]}`` for faces).
  For face recognition experiments, the given ``original_data`` might be either a color or a gray level image, and the returned ``data`` should be a :py:class:`numpy.ndarray` with 2D shape ``[height, width]`` containing floating point values, if possible (e.g. to be usable by all the algorithms).

  .. note::
    When the database does not provide annotations, the ``annotations`` parameter might be ``None``.

If your class returns data that is **not** of type :py:class:`numpy.ndarray`, you might need to overwrite further functions from :py:class:`facereclib.preprocessing.Preprocessor` that define the IO of your class:

* ``save_data(data, filename)``: Writes the given data (that has been generated using the ``__call__`` function of this class) to file.
* ``read_data(filename)``: Reads the preprocessed data from file.

By default, the original data is read by :py:func:`bob.io.base.load`.
Hence, data is given as :py:class:`numpy.ndarray`\s.
If you want to use a different IO for the original data (rarely useful...), you might want to overload:

* ``read_original_data(filename)``: Reads the original data from file.

If you plan to use a simple face cropping for facial image processing, you might want to derive your class from the :py:class:`facereclib.preprocessing.FaceCrop` class (you don't need to derive from :py:class:`facereclib.preprocessing.Preprocessor ` in this case).
In this case, just add a ``**kwargs`` parameter to your constructor, call the face crop constructor with these parameters: ``facereclib.preprocessing.FaceCrop.__init__(self, **kwargs)``, and call the ``self.face_crop(image, annotations)`` in your ``__call__`` function.
For an example of this behavior, you might have a look into the `facereclib.preprocessing.HistogramEqualization <file:../facereclib/preprocessing/HistogramEqualization.py>`_ class.


Feature Extractors
~~~~~~~~~~~~~~~~~~
Feature extractors should be derived from the :py:class:`facereclib.features.Extractor` class.
Your extractor class has to provide at least the functions:

* ``__init__(self, <parameters>)``: Initializes the feature extraction algorithm with the parameters it needs.
  Please call the base class constructor in this constructor, e.g. as ``facereclib.features.Extractor.__init__(self, ...)`` (there are more parameters to this constructor, see below).
* ``__call__(self, data) -> feature``: Extracts the feature from the given preprocessed data.
  By default, the returned feature should be a :py:class:`numpy.ndarray`.

If your features are not of type :py:class:`numpy.ndarray`, please overwrite the ``save_feature`` function to write features of other types.
Please also overwrite the function to read your kind of features:

* ``save_feature(self, feature, feature_file)``: Saves the feature (as returned by the ``__call__`` function) to the given file name.
* ``read_feature(self, feature_file) -> feature``: Reads the feature (as written by the ``save_feature`` function) from the given file name.

.. note::
  If your feature is of a class that contains and is written via a ``save(bob.io.base.HDF5File)`` method, you do not need to define a ``save_feature`` function.
  However, the ``read_feature`` function is required in this case.

If the feature extraction process requires to read a trained extractor model from file, simply overload the function:

* ``load(self, extractor_file)``: Loads the extractor from file.
  This function is called at least once before the ``__call__`` function is executed.

It is also possible to train the extractor model before it is used.
In this case, you have to do two things.
First, you have to overwrite the ``train`` function:

* ``train(self, image_list, extractor_file)``: Trains the feature extractor with the given list of images and writes the ``extractor_file``.

Second, you have to register this behavior in your ``__init__`` function by calling the base class constructor with more parameters: ``facereclib.features.Extractor.__init__(self, requires_training=True, ...)``.
Given that your training algorithm needs to have the training data split by identity, please use ``facereclib.features.Extractor.__init__(self, requires_training=True, split_training_images_by_client = True, ...)`` instead.


Recognition Algorithms
~~~~~~~~~~~~~~~~~~~~~~
Implementing your recognition algorithm should be as straightforward.
Simply derive your class from the :py:class:`facereclib.tools.Tool` class.

The :py:class:`facereclib.tools.Tool` constructor has the following options:

* ``performs_projection``: If set to ``True``, features will be projected using the ``project`` function.
  With the default ``False``, the ``project`` function will not be called at all.
* ``requires_projector_training``: If ``performs_projection`` is enabled, this flag specifies if the projector needs training.
  If ``True`` (the default), the ``train_projector`` function will be called.
* ``split_training_features_by_client``: If the projector training needs training images split up by client identity, please enable this flag.
  In this case, the ``train_projector`` function will receive a list of lists of features.
  If set to ``False`` (the default), the training features are given in one list.
* ``use_projected_features_for_enrollment``: If features are projected, by default (``True``) models be enrolled using the projected features.
  If your algorithm requires the original unprojected features to enroll the model, please set ``use_projected_features_for_enrollment=False``.
* ``requires_enroller_training``: Enables the enroller training.
  By default (``False``), no enroller training is performed, i.e., the ``train_enroller`` function is not called **even if you wrote it**.

* ``multiple_model_scoring``: The way to handle scoring when models store several features.
  Set this parameter to ``None`` when you implement your own functionality to handle models from several features (see below).
* ``multiple_probe_scoring``: The way to handle scoring when models store several features.
  Set this parameter to ``None`` when you handle scoring with multiple probes with your own ``score_for_multiple_probes`` function (see below).

A recognition tool has to have at least three functions:

* ``__init__(self, <parameters>)``: Initializes the face recognition algorithm with the parameters it needs.
  Please call the base class constructor in this constructor, e.g. as ``facereclib.tools.Tool.__init__(self, ...)`` (there are more parameters to this constructor, see above).
* ``enroll(self, enroll_features) -> model``: Enrolls a model from the given vector of features (this list usually contains features from several files of one subject) and returns it.
  The returned model should either be a :py:class:`numpy.ndarray` or an instance of a class that defines a ``save(bob.io.base.HDF5File)`` method.
  If neither of the two options are appropriate, you have to define a ``write_model`` function (see below).
* ``score(self, model, probe) -> value``: Computes a similarity or probability score that the given probe feature and the given model stem from the same identity.

  .. note::
    When you use a distance measure in your scoring function, and lower distances represents higher probabilities of having the same identity, please return the negative distance.

Additionally, your tool may need to project the features before they can be used for enrollment or recognition. In this case, simply overwrite (some of) the function(s):

* ``train_projector(self, train_features, projector_file)``: Uses the given list of features and writes the ``projector_file``.

  .. note::
    If you write this function, please assure that you use both ``performs_projection=True`` and ``requires_projector_training=True`` (for the latter, this is the default, but not for the former) during the base class constructor call in your ``__init__`` function.
    If you need the training data to be sorted by clients, please use ``split_training_features_by_client=True`` as well.
    Please also assure that you overload the ``project`` function.

* ``load_projector(self, projector_file)``: Loads the projector from the given file.
  This function is always called before the ``project``, ``enroll``, and ``score`` functions are executed.
* ``project(self, feature) -> feature``: Projects the given feature and returns the projected feature, which should either be a :py:class:`numpy.ndarray` or an instance of a class that defines a ``save(bob.io.base.HDF5File)`` method.

  .. note::
    If you write this function, please assure that you use ``performs_projection=True`` during the base class constructor call in your ``__init__`` function.

And once more, if the projected feature is not of type ``numpy.ndarray``, overwrite the methods:

* ``save_feature(feature, feature_file)``: Writes the feature (as returned by the ``project`` function) to file.
* ``read_feature(feature_file) -> feature``: Reads and returns the feature (as written by the ``write_feature`` function).

Some tools also require to train the model enroller.
Again, simply overwrite the functions:

* ``train_enroller(self, training_features, enroller_file)``: Trains the model enrollment with the list of lists of features and writes the ``enroller_file``.

  .. note::
    If you write this function, please assure that you use ``requires_enroller_training=True`` during the base class constructor call in your ``__init__`` function.

* ``load_enroller(self, enroller_file)``: Loads the enroller from file.
  This function is always called before the ``enroll`` and ``score`` functions are executed.


By default, it is assumed that both the models and the probe features are of type :py:class:`numpy.ndarray`.
If your ``score`` function expects models and probe features to be of a different type, you should overwrite the functions:

* ``save_model(self, model, model_file)``: writes the model (as returned by the ``enroll`` function)
* ``read_model(self, model_file) -> model``: reads the model (as written by the ``write_model`` function) from file.
* ``read_probe(self, probe_file) -> feature``: reads the probe feature from file.

  .. note::
    In many cases, the ``read_feature`` and ``read_probe`` functions are identical (if both are present).

Finally, the :py:class:`facereclib.tools.Tool` class provides default implementations for the case that models store several features, or that several probe features should be combined into one score.
These two functions are:

* ``score_for_multiple_models(self, models, probe)``: In case your model store several features, **call** this function to compute the average (or min, max, ...) of the scores.
* ``score_for_multiple_probes(self, model, probes)``: By default, the average (or min, max, ...) of the scores for all probes are computed. **Overwrite** this function in case you want different behavior.



Executing experiments with your classes
---------------------------------------
Finally, executing experiments using your database, preprocessor, feature extraction, and/or recognition tool should be as easy as using the tools that are already available.
Nonetheless, it might be a good idea to first run the experiments locally (i.e., calling the ``./bin/faceverify.py -vvv`` without the ``--grid`` option) to see if your functions do work and do provide expected results.
For this, it might also be a good idea to use a small image database, like ``--database atnt``.

.. note::
  In case you implement a preprocessor, which applies face alignment, using the ``atnt`` database is not a good idea since it does not contain annotations.


Adding Unit Tests
-----------------
To make sure that your piece of code it working properly, you should add a test case for your class.
The FaceRecLib, as well as Bob_, rely on `nose tests <http://pypi.python.org/pypi/nose>`_ to run the unit tests.
To implement a unit test for your contribution, you simply can create a python file with a name containing 'test' in your package.
In the FaceRecLib, these files are located in `facereclib/tests <file:../facereclib/tests>`_.

In the test file, please write a test class that derives from ``unittest.TestCase``.
Any function name containing the string ``test`` will be automatically found and executed when running ``./bin/nosetests``.
In your test function, please assure that all aspects of your contribution are thoroughly tested and that all test cases pass.
Also remember that your tests need to run on different machines with various operating systems, so don't test floating point values for equality.


.. _configuration-files:

Adding Configuration Files
--------------------------
After your code is tested, you should provide a configuration file for your algorithm.
A configuration file basically consists of a constructor call to your new class with a useful (yet not necessarily optimized) set of parameters.
Depending on your type of contribution, you should write a line like:

* ``database = facereclib.databases.<YourDatabase>(<YourParameters>)``
* ``preprocessor = facereclib.preprocessing.<YourPreprocessor>(<YourParameters>)``
* ``feature_extractor = facereclib.features.<YourExtractor>(<YourParameters>)``
* ``tool = facereclib.tools.<YourAlgorithm>(<YourParameters>)``

and save the configuration file into the according sub-directory of `facereclib/configurations <file:../facereclib/configurations>`_.


.. _register-resources:

Registering your Code as a Resource
-----------------------------------
Now, you should be able to register this configuration file as a resource, so that you can use the configuration from above by a simple ``<shortcut>`` of your choice.
Please open the `setup.py <file:../setup.py>`_ file in the base directory of your satellite package and edit the ``entry_points`` section.
Depending on your type of algorithm, you have to add:

* ``'facereclib.database': [ '<your-database-shortcut> = <your-database-configuration>.database' ]``
* ``'facereclib.preprocessor': [ '<your-preprocessor-shortcut> = <your-preprocessor-configuration>.preprocessor' ]``
* ``'facereclib.feature_extractor': [ '<your-extractor-shortcut> = <your-extractor-configuration>.feature_extractor' ]``
* ``'facereclib.tool': [ '<your-recognition-algorithm-shortcut> = <your-algorithm-configuration>.tool' ]``

After re-running ``./bin/buildout``, your new resource should be listed in the output of ``./bin/resources.py``.


.. include:: links.rst
