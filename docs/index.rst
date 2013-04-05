.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. FaceRecLib documentation master file, created by
   sphinx-quickstart on Thu Sep 20 11:10:55 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================================
 Welcome to |project|'s documentation!
=======================================

The |project| is an open source platform that is designed to run comparable and reproducible face recognition experiments.
To design a face recognition experiment, one has to choose:

* an image databases and its according protocol,
* face detection and image preprocessing algorithms,
* the type of features to extract from the face,
* the face recognition algorithm to employ, and
* the way to evaluate the results

For any of these parts, several different types are implemented in the |project|, and basically any combination of the five parts can be executed.
For each type, several meta-parameters can be tested.
This results in a nearly infinite amount of possible face recognition experiments that can be run using the current setup.
But it is also possible to use your own database, preprocessing, feature type, or face recognition algorithm and test this against the baseline algorithms implemented in the |project|.

If you are interested, please continue reading:


.. toctree::
   :maxdepth: 2


   installation
   baselines
   experiments
   specialized
   contribute
   satellite
   references

==========================
|project| reference manual
==========================

.. toctree::
   :maxdepth: 2

   manual_databases
   manual_preprocessors
   manual_features
   manual_tools


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

.. include: links.rst
