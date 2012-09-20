.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

Installation instructions
=========================

.. TODO::

  Correct the installation instructions as soon as you have uploaded it to github and pypi.

To install the |project|, please check the latest version of it via:

.. code-block:: sh

  $ git clone /idiap/group/torch5spro/sandboxes/facereclib2.git
  $ cd facereclib2

For the |project| to work, it requires `Bob`_ to be installed.
At Idiap, you can either have your local Bob installation or use the global one located at:

::

  > /idiap/group/torch5spro/nightlies/last/install/<VERSION>-release

where <VERSION> is your operating system version.

The |project| project is based on the `BuildOut`_ python linking system.
If you want to use another version of Bob than the nightlies, you have to modify the delivered *buildout.cfg* by specifying the path to your Bob installation.

Afterwards, execute the buildout script by typing:

.. code-block:: sh

  $ /remote/filer.gx/group.torch5spro/nightlies/externals/v3/ubuntu-10.04-x86_64/bin/python bootstrap.py
  $ bin/buildout


Running the tests
-----------------

Each element of the |project| has a test to assure that it is executing properly.


.. include:: links.rst
