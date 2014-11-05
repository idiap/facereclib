.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _facereclib:

========================================
 Welcome to FaceRecLib's documentation!
========================================

The FaceRecLib is an open source tool that is designed to run comparable and reproducible face recognition experiments.
To design a face recognition experiment, one has to choose:

* an image databases and its according protocol,
* face detection and image preprocessing algorithms,
* the type of features to extract from the face,
* the face recognition algorithm to employ, and
* the way to evaluate the results

For any of these parts, several different types are implemented in the FaceRecLib, and basically any combination of the five parts can be executed.
For each type, several meta-parameters can be tested.
This results in a nearly infinite amount of possible face recognition experiments that can be run using the current setup.
But it is also possible to use your own database, preprocessing, feature type, or face recognition algorithm and test this against the baseline algorithms implemented in the FaceRecLib.

For example, we created wrapper classes for the `CSU Face Recognition Resources`_, which can be found in the xfacereclib.extension.CSU_ satellite package, including installation instructions for the CSU toolkit.

If you are interested, please continue reading:


===========
Users Guide
===========

.. toctree::
   :maxdepth: 2

   installation
   baselines
   experiments
   evaluate
   specialized
   contribute
   satellite
   references

================
Reference Manual
================

.. toctree::
   :maxdepth: 2

   manual_databases
   manual_preprocessors
   manual_features
   manual_tools
   manual_utils


ToDo-List
=========

This documentation is still under development.
Here is a list of things that needs to be done:

.. todolist::


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: links.rst
