.. vim: set fileencoding=utf-8 :
.. Manuel Guenther <Manuel.Guenther@idiap.ch>
.. Mon 23 04 2012

========================================
Implementing your own features and tools
========================================

The FaceRecLib module is specifically designed to be as flexible as possible while trying to keep things simple. FaceRecLib uses python to implement algorithms. It is file-based so any algorithm can implement their own way of reading and writing features or models. Algorithm configurations are read from configuration files, so it should be easy to test different parameters of your algorithms without modifying the code.

To implement your own features or tools, simply follow the examples that are already in the FaceRecLib. In the following sections there will be an overview of the functions that need to be implemented.

Image preprocessing
-------------------
Classes that do image preprocessing have to have at least the two functions:

* ``__init__(self, config)``: where ``config`` is the image preprocessing configuration read from file (usually one in *config/features*)
* ``__call__(self, input_file, output_file, eye_pos)``: normalizes the face given the vector of eye positions in the order ``[re_x, re_y, le_x, le_y]`` and preprocesses it afterward. The given ``input_file`` must be read beforehand, maybe using ``bob.io`` functionality. The written ``output_image`` should be a numpy.ndarray, if possible (e.g. to be usable by all the algorithms) as 2D with shape ``[height, width]``.

When the database does not provide eye positions, the ``eye_pos`` parameter might be ``None``. In this case either the images are already aligned to the eye positions, or your class is expected to perform face detection (this is depending on the database, please assure that your preprocessor class and your database fit together).

Please also add a configuration file in *config/features* that uses your class and that includes some default configuration of your algorithm. Please note that the configuration file of the image preprocessing is shared with the configuration file of the feature extraction stage.

Feature extraction
------------------
In the feature extraction stage, the classes have to provide at least the functions:

* ``__init__(self, config)``: where ``config`` is the feature extraction configuration read from file  (usually one in *config/features*)
* ``__call__(self, image) -> feature``: extracts the feature from the given preprocessed image. The returned feature should be a numpy.ndarray. 

If your features are not of type numpy.ndarray, please add a ``save_feature`` function (see below) to write features of other types (note that the currently implemented tools require numpy.ndarray features). Please also provide a function to read your kind of features:

* ``save_feature(self, feature, feature_file)``: saves the feature (as returned by the ``__call__`` function) to the given filename
* ``read_feature(self, feature_file) -> feature``: reads the feature (as written by the ``save_feature`` function) from the given filename

If the feature extraction process requires a trained extractor model, simply define the function:

* ``load(self, extractor_file)``: loads the extractor from file; this function is always called before the ``__call__`` function is executed.

It is also possible to train the model before it is used. For that purpose, simply define the function:

* ``train(self, image_list, extractor_file)``: trains the feature extraction with the given dictionary of image filenames ``file_id -> image_filename`` and writes the ``extractor_file``

or (given that your training algorithm needs to have the training data split by identity):

* put the line ``self.use_training_images_sorted_by_identity = True`` into your ``__init__`` function 
* ``train(self, image_list, extractor_file)``: trains the feature extraction with the two layered dictionary structure ``person_id -> {file_id -> image_filename}`` and writes the ``extractor_file``

Finally, add a configuration file to *config/features*. Please note that the configuration file of the feature extraction is shared with the configuration file of the image preprocessing stage.


Tools
-----
Implementing your face recognition tool should be as straightforward. A face recognition tool has to have at least three functions:

* ``__init__(self, config)``: where ``config`` is the tool configuration read from file  (usually one in *config/tools*)
* ``enrol(self, enrol_features) -> model``: enrolls a model from the given vector of features (which usually stem from one identity). The returned model should either be a numpy.ndarray or an instance of a class that defines a ``save(bob.io.HDF5File)`` method. If neither of the two options are appropriate, you have to define a ``write_model`` function (see below).
* ``score(self, model, probe) -> value``: computes a similarity or probability score that the given probe feature and the given model include the same identity

Additionally, your tool may need to project the features before they can be used for enrollment or recognition. In this case, simply define (some of) the function(s):

* ``train_projector(self, train_files, projector_file)``: uses the given dictionary of image filenames ``file_id -> image_filename`` and writes the ``projector_file``
* ``load_projector(self, projector_file)``: loads the projector from file; this function is always called before the ``project``, ``enrol``, and ``score`` functions are executed.
* ``project(self, feature) -> feature``: projects the given feature.

Again, if training data need to be sorted by identity:

* Add the line: ``self.use_training_features_sorted_by_identity = True`` to your ``__init__`` function 
* ``train_projector(self, train_files, projector_file)``: trains the projector with the two layered dictionary structure ``person_id -> {file_id -> image_filename}`` and writes the ``projector_file``

Some tools also require to train the model enrollment. Again, simply add the functions:

* ``train_enroler(self, training_features, enroler_file)``: trains the model enrollment with the two layered dictionary structure ``person_id -> {file_id -> image_filename}`` and writes the ``enroler_file``
* ``load_enroler(self, enroler_file)``: loads the enroler from file; this function is always called before the ``enrol`` and ``score`` functions are executed.

By default, the features that will be passed to the ``enrol`` (and ``train_enroler``) function(s) are the projected features (if the tool provides a ``project`` function) or the unprojected features (i.e., the result of the feature extraction stage) otherwise. If your tool defines a ``project`` function, but your enrollment requires unprojected features, simply:

* Add the line: ``self.use_unprojected_features_for_model_enrol = True`` to your ``__init__`` function.

Usually, projected features are of type numpy.ndarray, and models are either of the same type, or of any class that defines a ``save(bob.io.HDF5File)`` method. If your projected features and your models are of a different data type, you might want to specify:

* ``save_feature(self, feature, feature_file)``: saves the feature (as returned by the ``project`` function) to file
* ``save_model(self, model, model_file)``: saves the model (as returned by the ``enrol`` function) to file

By default, it is assumed that both the models and the probe features are numpy.ndarrays. If your ``score`` function expects models and probe features to be of a different type, you might add the functions:

* ``read_model(self, model_file) -> model``: reads the model from file
* ``read_probe(self, probe_file) -> feature``: reads the probe feature from file

Add the end, please provide a configuration file for your tool in *config/tools*.


Executing experiments with your classes
---------------------------------------
Finally, executing experiments using your image preprocessing, feature extraction, and/or recognition tool should be identical to the tools that are already available. Nonetheless, it might be a good idea to first run the experiments locally (i.e., calling the *bin/faceverify_zt.py* without the ``--grid`` option) to see if your functions do work and do provide expected results. It might also be a good idea to use a small image database, like *config/database/banca_P.py*.


