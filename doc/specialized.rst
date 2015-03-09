.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Fri Oct 26 17:05:40 CEST 2012

.. _specialized-scripts:

=============================================
Running Experiments using Specialized Scripts
=============================================

.. _parameter-tests:

Testing Configurations of an Algorithm
--------------------------------------
Sometimes, configurations of algorithms are highly dependent on the database or even the employed protocol.
Additionally, configuration parameters depend on each other.
The FaceRecLib provides a relatively simple set up that allows to test different configurations in the same task.
For this, the ``./bin/parameter_test.py`` script can be employed.
This script executes a configurable series of experiments, which reuse data as far as possible.

The Configuration File
~~~~~~~~~~~~~~~~~~~~~~
The most important parameter to the ``./bin/parameter_test.py`` is the ``--configuration-file``.
In this configuration file it is specified, which parameters of which part of the algorithms will be tested.
An example for a configuration file can be found in the test scripts: `facereclib/tests/scripts/parameter_Test.py <file:../facereclib/tests/scripts/parameter_Test.py>`_.
The configuration file is a common python file, which can contain certain variables:

1. ``preprocessor =``
2. ``feature_extractor =``
3. ``tool =``
4. ``replace =``
5. ``requirement =``
6. ``imports =``

The variables from 1. to 3. usually contain constructor calls for classes of :ref:`preprocessors`, :ref:`extractors` and :ref:`algorithms`, but also registered :ref:`Resources <managing-resources>` can be used.
For any of the parameters of the classes, a `placeholder` can be put.
By default, these place holders start with a # character, followed by a digit or character.
The variables 1. to 3. can also be overridden by the command line options ``--preprocessing``, ``--features`` and ``--tool`` of the ``./bin/parameter_test.py`` script.

The ``replace`` variable has to be set as a dictionary.
In it, you can define with which values your place holder key should be filled, and in which step of the tool chain execution this should happen.
The steps are ``'preprocessing'``, ``'extraction'``, ``'projection'``, ``'enrollment'`` and ``'scoring'``.
For each of the steps, it can be defined, which placeholder should be replaced by which values.
To be able to differentiate the results later on, each of the replacement values is bound to a directory name.
The final structure looks somewhat like that:

.. code-block:: python

  replace = {
      step1 : {
          '#a' : {
              'Dir_a1' : 'Value_a1',
              'Dir_a2' : 'Value_a2'
           },

          '#b' : {
              'Dir_b1' : 'Value_b1',
              'Dir_b2' : 'Value_b2'
          }
      },

      step2 : {
          '#c' : {
              'Dir_c1' : 'Value_c1',
              'Dir_c2' : 'Value_c2'
          }
      }
  }

Of course, more than two values can be selected.
Additionally, tuples of place holders can be defined, in which case always the full tuple will be replaced in one shot.
Continuing the above example, it is possible to add:

.. code-block:: python

  ...
      step3 : {
          ('#d','#e') : {
              'Dir_de1' : ('Value_d1', 'Value_e1'),
              'Dir_de2' : ('Value_d2', 'Value_e2')
          }
      }

Note that **all possible combinations** of the configuration parameters are tested, which might result in a **huge number of executed experiments**.
Some combinations of parameters might not make any sense.
In this case, a set of requirements on the parameters can be set, using the ``requirement`` variable.
In the requirements, any string including any placeholder can be put that can be evaluated using pythons ``eval`` function:

.. code-block:: python

  requirement = ['#a > #b', '2*#c != #a', ...]

Finally, if any of the classes or variables need to import a certain python module (other than the ``facereclib``), it needs to be declared in the ``imports`` variable.
If you, e.g., test, which ``scipy`` distance function works best for your features, please add the imports (and don't forget the ``facereclib`` in case you use its tools):

.. code-block:: python

  imports = ['scipy', 'facereclib']

A complete working example, where the image resolution and LGBPHS distance function are tested, can be found in `facereclib/tests/scripts/parameter_Test.py <file:../facereclib/tests/scripts/parameter_Test.py>`_.


Further Command Line Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``./bin/parameter_test.py`` script has a further set of command line options.

- The ``--database`` and the ``--protocol`` define, which database and (optionally) which protocol should be used.
- The ``--sub-directory`` is similar to the one in the ``./bin/faceverify.py``, see :ref:`required`.
- The ``--preprocessing``, ``--features`` and ``--tool`` can be used to override the ``preprocessor``, ``feature_extractor`` and ``tool`` fields in the configuration file (in which case the configuration file does not need to contain these variables).
- The ``--grid`` option can select the SGE_ configuration (if not selected, **all** experiments will be run sequentially on the local machine).
- The ``--preprocessed-data-directory`` can be used to select a directory of previously preprocessed data. This should not be used in combination with testing different preprocessing parameters.
- The ``--grid-database-directory`` can be used to select another directory, where the `submitted.sql3` files will be stored.
- The ``--write-commands`` directory can be selected to write the executed commands into (this is useful in case some experiments fail and need to be rerun).
- The ``--dry-run`` flag should always be used before the final execution to see if the experiment definition works as expected.
- The ``--skip-when-existent`` flag will only exexute the experiments that have not yet finished (i.e., where the resulting score files are not produced yet).
- Finally, additional options might be sent to the ``./bin/faceverify.py`` script directly. These options might be put after a ``--`` separation.


Evaluation of Results
~~~~~~~~~~~~~~~~~~~~~

To evaluate a series of experiments, a special script iterates through all the results and computes EER on the development set and HTER on the evaluation set, for both the **nonorm** and the **ztnorm** directories.
Simply call:

.. code-block:: sh

  $ ./bin/collect_results.py --directory [result-base-directory] --sort

This will iterate through all result files found in [result-base-directory] and sort the results according to the EER on the development set (the sorting criterion can be modified using the ``--criterion`` keyword).


Databases with Special Evaluation Protocols
-------------------------------------------
Some databases provide special evaluation protocols which require a more complicated experiment design.
For these databases, different scripts are provided.
These databases are:

The LFW Database
~~~~~~~~~~~~~~~~
For the :ref:`Labeled Faces in the Wild <bob.db.lfw>` (LFW) database, there is another script to calculate the experiments, strictly following the LFW protocols by computing the classification performance on `view1` and/or `view2`.
The final result of the LFW experiment is, hence, a text file (``--result-file``) containing the single results for ``view1`` and the 10 folds ``fold1`` ... ``fold10`` of ``view2``, as well as the final average and standard deviation of all folds.
In principle, the ``./bin/faceveryfy.py`` could be used as well, without having the classification performance.

The parameters of the ``./bin/faceverify_lfw.py`` script are mostly similar to the ``./bin/faceverify.py`` script as explained in :ref:`experiments`.
A few exceptions are that the default database is ``lfw`` and the parts belonging to the ZT score normalization are missing.
Additionally, instead of the ``--protocol`` option, the ``--views`` option is available, which by default executes only on ``view1``.

The GBU Database
~~~~~~~~~~~~~~~~

There is another script ``./bin/faceverify_gbu.py`` that executes experiments on the :ref:`Good, Bad, and Ugly <bob.db.gbu>` (GBU) database.
In principle, most of the parameters from above can be used.
One violation is that instead of the ``--models-directories`` option is replaced by only ``--model-directory``.

When running experiments on the GBU database, the default GBU protocol (as provided by NIST_) is used.
Hence, training is performed on the special Training set, and experiments are executed using the Target set as models (using a single image for model enrollment) and the Query set as probe.

The GBU protocol does not specify T-Norm-models or Z-Norm-probes, nor it splits off development and test set.
Hence, only a single score file is generated, which might later on be converted into an ROC curve using Bob functions.


.. include:: links.rst
