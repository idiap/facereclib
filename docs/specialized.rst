.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Fri Oct 26 17:05:40 CEST 2012

.. _specialized-scripts:

=============================================
Running experiments using specialized scripts
=============================================

.. _parameter-tests:

Testing parameters of your algorithm
------------------------------------

.. TODO::

  Write the documentation of the parameter testing script.


Databases with special evaluation protocols
-------------------------------------------

Some databases provide special evaluation protocols which require a more complicated experiment design.
For these databases, different scripts are provided.
These databases are:

The LFW database
~~~~~~~~~~~~~~~~
For the `Labeled Faces in the Wild` (LFW) database, there is another script to calculate the experiments, strictly following the LFW protocols.

.. TODO::

  Write the documentation of the LFW script.


The GBU database
~~~~~~~~~~~~~~~~

.. TODO::

  remove this section since this script is outdated.

There is another script *bin/faceverify_gbu.py* that executes experiments on the Good, Bad, and Ugly (GBU) database.
In principle, most of the parameters from above can be used.
One violation is that instead of the ``--models-directories`` option is replaced by only ``--model-directory``.

When running experiments on the GBU database, the default GBU protocol (as provided by `NIST`_) is used.
Hence, training is performed on the special Training set, and experiments are executed using the Target set as models (using a single image for model enrollment) and the Query set as probe.

The GBU protocol does not specify T-Norm-models or Z-Norm-probes, nor it splits off development and test set.
Hence, only a single score file is generated, which might later on be converted into an ROC curve using Bob functions.


.. include:: links.rst
