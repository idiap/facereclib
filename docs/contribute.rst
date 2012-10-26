.. vim: set fileencoding=utf-8 :
.. Manuel Guenther <Manuel.Guenther@idiap.ch>
.. Mon 23 04 2012

========================================
Implementing your own features and tools
========================================

The |project| module is specifically designed to be as flexible as possible while trying to keep things simple.
|project| uses python to implement algorithms.
It is file based so any algorithm can implement their own way of reading and writing features or models.
Algorithm configurations are read from configuration files, so it should be easy to test different parameters of your algorithms without modifying the code.

To implement your own preprocessing, features, or tools, simply follow the examples that are already in the |project|.
In the following sections there will be an overview of the functions that need to be implemented.

The |project| is designed in a way that useful default functionalities are executed.
If you want/need to have a different behavior, you can simply add functions to your classes that will be detected and executed automatically.


Creating your own satellite package
-----------------------------------

.. TODO:
  Explain how to create a satellite package after you have successfully done that.


Implementing your own functions
-------------------------------


Image databases
~~~~~~~~~~~~~~~
If you have your own database of images that you want to execute the recognition experiments on, you should first check if you could use the ``xbob.db.faceverif_fl`` database interface by defining appropriate file lists for the training set, the model set, and the probes.
If you can do this, just write your own configuration file that uses the ``facereclib.databases.DatabaseXBob`` and ``facereclib.databases.DatabaseXBobZT`` interface (see :ref:`databases` for details).
In most of the cases, the ``xbob.db.faceverif_fl`` should be sufficient to run experiments.

In case you want to have a more complicated interface to your database, you are welcome to write your own database wrapper class.
In this case, you have to derive your class from the ``facereclib.databases.Database``, and provide the following functions:

* ``__init__(self, <your-parameters>, **kwargs)``: Constructor of your database interface.
  Please call the base class constructor, providing all the required parameters (see :ref:`databases`), e.g. by ``facereclib.databases.Database.__init__(self, **kwargs)``.
* ``all_files(self)``: Returns a list of all ``facereclib.databases.File`` objects of the database.
  The list needs to be sorted by the file id (you can use the ``self.sort(files)`` function for sorting).
* ``training_files(self, step, arrange_by_client = False)``: A sorted list of the ``facereclib.databases.File`` objects that is used for training.
  If ``arrange_by_clients`` is enabled, you might want to use the ``self.arrange_by_client(files)`` function to perform the job.
* ``model_ids(self, group = 'dev'): The ids for the models (usually, there is only one model per client and, thus, you can simply use the client ids) for the given group.
  Usually, providing ids for the group ``'dev'`` should be sufficient.
* ``client_id_from_model_id(self, model_id)``: Returns the client id for the given model id.
* ``enroll_files(self, model_id, group='dev')``: Returns the list of model ``facereclib.databases.File`` objects for the given model id.
* ``probe_files(self, model_id=None, group='dev')``: Returns the list of probe files, the given model_id should be compared with.
  Usually, all probe files are compared with all model files.
  In this case, you can just ignore the ``model_id``.
  If the ``model_id`` is ``None``, this function is supposed to return *all* probe files for all models of the given group.

Additionally, you can define more lists that can be used for ZT score normalization.
In this case, derive you class from ``facereclib.databases.DatabaseZT`` instead, and additionally overwrite the following functions:

* ``t_model_ids(self, group = 'dev')``: The ids for the T-Norm models for the given group.
* ``t_enroll_files(self, model_id, group='dev')``: Returns the list of model ``facereclib.databases.File`` objects for the given T-Norm model id.
* ``z_probe_files(self, group='dev')``: Returns the list of Z-probe files, with which all the models and T-Norm models are compared.

.. note:
  For a proper face recognition protocol, the identities from the models and the T-Norm models, as well as the Z-probes should be different.


Image preprocessors
~~~~~~~~~~~~~~~~~~~
All image preprocessing classes should be derived from the ``facereclib.preprocessing.Preprocessor`` class.
In your class,


* ``__init__(self, <parameters>)``: Initializes the image preprocessing algorithm with the parameters it needs.
  Please call the base class constructor in this constructor, e.g. as ``facereclib.preprocessing.Preprocessor.__init__(self)``.
* ``__call__(self, input_image, annotations) -> image``: geometrically normalizes the image given the dictionary of annotations (e.g. ``{'reye' : [re_y, re_x], 'leye': [le_y, le_x]}``) and preprocesses it.
  The given ``input_image`` might be either a color or a gray level image.
  The returned ``image`` should be a numpy.ndarray with 2D shape ``[height, width]`` containing floating point values, if possible (e.g. to be usable by all the algorithms).

  .. note::
    When the database does not provide eye positions, the ``annotations`` parameter might be ``None``.
    In this case either the images are already aligned to the eye positions, or your class is expected to perform face detection (this is depending on the database, please assure that your preprocessor class and your database fit together).

If your class returns images that are **not** of type numpy.ndarray, you need to overwrite further functions from ``facereclib.preprocessing.Preprocessor`` that define the IO of your class:

* ``save_image(image, filename)``: Writes the given image (that has been preprocessed using the ``__call__`` function of this class) to file.
* ``read_image(filename)``: Reads the preprocessed image from file.
* ``read_original_image(filename)``: Reads the original image data from file (rarely used).


If you plan to use a simple face cropping, you might want to derive your class from the ``facereclib.preprocessing.FaceCrop`` class (you don't need to derive from ``facereclib.preprocessing.Preprocessor`` in this case).
In this case, just add a ``**kwargs`` parameter to your constructor, call the face crop constructor with these parameters: ``facereclib.preprocessing.FaceCrop.__init__(self, **kwargs)``, and call the ``self.face_crop(image, annotations)`` in your ``__call__` function.
As an example you should have a look into the ``facereclib.preprocessing.TanTriggs`` class.


Feature extractors
~~~~~~~~~~~~~~~~~~
Feature extractors should be derived from the ``facereclib.features.Extractor`` class.
Your extractor class has to provide at least the functions:

* ``__init__(self, <parameters>)``: Initializes the feature extraction algorithm with the parameters it needs.
  Please call the base class constructor in this constructor, e.g. as ``facereclib.features.Extractor.__init__(self)`` (there are more parameters to this constructor, see below).
* ``__call__(self, image) -> feature``: Extracts the feature from the given preprocessed image.
  The returned feature should be a numpy.ndarray.

If your features are not of type numpy.ndarray, please overwrite the ``save_feature`` function to write features of other types.
Please also overwrite the function to read your kind of features:

* ``save_feature(self, feature, feature_file)``: Saves the feature (as returned by the ``__call__`` function) to the given file name.
* ``read_feature(self, feature_file) -> feature``: Reads the feature (as written by the ``save_feature`` function) from the given file name.

.. note::
  If your feature is of a class that contains and is written via a ``save(bob.io.HDF5File)`` method, you do not need to define a ``save_feature`` function.
  Remember: the ``read_feature`` function is required in this case.

If the feature extraction process requires to read a trained extractor model from file, simply define the function:

* ``load(self, extractor_file)``: Loads the extractor from file.
  This function is always called before the ``__call__`` function is executed.

It is also possible to train the extractor model before it is used.
For that purpose, simply overwrite the function:

* ``train(self, image_list, extractor_file)``: Trains the feature extractor with the given list of images and writes the ``extractor_file``.

or (given that your training algorithm needs to have the training data split by identity):

* ``train(self, image_list, extractor_file)``: trains the feature extraction with the two layered list of images and writes the ``extractor_file``.
  In this case, the first index into the image_list is the person identity, while the second index is the image for that identity.

and register your function by calling the appropriate base class constructor in your constructor, e.g. ```facereclib.features.Extractor.__init__(self, requires_training = True, split_training_images_by_client = True)``.



Face recognition algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implementing your face recognition tool should be as straightforward.
Simply derive your class from the ``facereclib.tools.Tool`` class.
A face recognition tool has to have at least three functions:

* ``__init__(self, <parameters>)``: Initializes the face recognition algorithm with the parameters it needs.
  Please call the base class constructor in this constructor, e.g. as ``facereclib.tools.Tool.__init__(self)`` (there are more parameters to this constructor, see below).
* ``enroll(self, enroll_features) -> model``: Enrolls a model from the given vector of features (which usually stem from one identity) and returns it.
  The returned model should either be a numpy.ndarray or an instance of a class that defines a ``save(bob.io.HDF5File)`` method.
  If neither of the two options are appropriate, you have to define a ``write_model`` function (see below).
* ``score(self, model, probe) -> value``: Computes a similarity or probability score that the given probe feature and the given model stem from the same identity.

  .. note::
    When you use a distance measure in your scoring function, and lower distances represents higher probabilities of having the same identity, please return the negative distance.

Additionally, your tool may need to project the features before they can be used for enrollment or recognition. In this case, simply overwrite (some of) the function(s):

* ``train_projector(self, train_features, projector_file)``: Uses the given list of features and writes the ``projector_file``.
* ``load_projector(self, projector_file)``: Loads the projector from the given.
  This function is always called before the ``project``, ``enroll``, and ``score`` functions are executed.
* ``project(self, feature) -> feature``: Projects the given feature and returns the projected feature, which should either be a numpy.ndarray or an instance of a class that defines a ``save(bob.io.HDF5File)`` method.

Again, if training data need to be sorted by identity:

* Define ``train_projector(self, train_features, projector_file)``: Trains the projector with the two layered list of features and writes the ``projector_file`` (see the training of the feature extractors).

And once more, if the feature is not of type ``numpy.ndarray``, overwrite the methods:

* ``write_feature(feature, feature_file)``: Writes the feature (as returned by the ``project`` function) to file.
* ``read_feature(feature_file) -> feature``: Reads and returns the feature (as written by the ``write_feature`` function).

Some tools also require to train the model enrollment.
Again, simply overwrite the functions:

* ``train_enroller(self, training_features, enroller_file)``: Trains the model enrollment with the two layered list of features and writes the ``enroller_file``.
* ``load_enroller(self, enroller_file)``: Loads the enroller from file.
  This function is always called before the ``enroll`` and ``score`` functions are executed.

To register the ``train_projector`` and/or the ``train_enroller`` functions, you have to call the base class constructor.
The ``facereclib.tools.Tool`` constructor has the following options:

* ``performs_projection``: If set to ``True``, features will be projected using the ``project`` function.
  With the default ``False`` the ``project`` function will not be called at all.
* ``requires_projector_training``: If ``performs_projection`` is enabled, this flag specifies if the projector needs training.
  If ``True`` (the default), the ``train_projector`` function will be called.
* ``split_training_features_by_client``: If the projector training needs training images split up by client identity, please enable this flag.
  If set to ``False`` (the default), the training features are given in one list.
* ``use_projected_features_for_enrollment``: If features are projected, by default (``True``) models be enrolled using the projected features.
  If your algorithm requires the original unprojected features to enroll the model, please set ``use_projected_features_for_enrollment = False``.
* ``requires_enroller_training``: Enables the enroller training.
  By default (``False``), no enroller training is performed.

By default, it is assumed that both the models and the probe features are numpy.ndarrays. If your ``score`` function expects models and probe features to be of a different type, you should overwrite the functions:

* ``read_model(self, model_file) -> model``: reads the model from file.
* ``read_probe(self, probe_file) -> feature``: reads the probe feature from file.

  .. note::
    In many cases, the ``read_feature`` and ``read_probe`` functions are identical (if both are present).



Executing experiments with your classes
---------------------------------------
Finally, executing experiments using your image preprocessing, feature extraction, and/or recognition tool should be as easy as using the tools that are already available.
Nonetheless, it might be a good idea to first run the experiments locally (i.e., calling the ``bin/faceverify.py -vvv`` without the ``--grid`` option) to see if your functions do work and do provide expected results.
For this, it might also be a good idea to use a small image database, like *atnt*.


Adding tests
------------
To make sure that your peace of code it working properly, you should add a test case for your class.

.. TODO:
  explain how to write tests properly.


.. _configuration-files:

Adding configuration files
--------------------------
After your code is tested, you should provide a configuration file for your algorithm.
A configuration file basically consists of a constructor call to your new class with a useful (yet not necessarily optimized) set of parameters.


.. _register-resources:

Registering your code as a resource
-----------------------------------
To be able to register this configuration file as a resource, it has to be in a the directory that is part of the module of your satellite package.
To register it, please open the **setup.py** file in the base directory of your satellite package and edit the ``entry_points`` section.
Depending on your type of algorithm, you have to add:

* ``'facereclib.database': [ '<your-database-shortcut>' : '<your-database-configuration>.database' ]``
* ``'facereclib.preprocessor': [ '<your-preprocessor-shortcut>' : '<your-preprocessor-configuration>.preprocessor' ]``
* ``'facereclib.feature_extractor': [ '<your-extractor-shortcut>' : '<your-extractor-configuration>.feature_extractor' ]``
* ``'facereclib.tool': [ '<your-recognition-algorithm-shortcut>' : '<your-algorithm-configuration>.tool' ]``

After re-running ``bin/buildout``, your new resource should be listed in the output of ``bin/resources.py``.


Contributing your code
----------------------
When you invented a completely new type of image preprocessing, features, or face recognition algorithm, and you want to share your result with the world, you are highly welcome **and encouraged** to do so.
Please make sure that every part of your code is documented and tested.

.. TODO:
  Add documentation how to upload a new satellite package to github and/or create a PyPI package from your satellite package.


.. include:: links.rst
