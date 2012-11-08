==============================
 The Face Recognition Library
==============================

Welcome to the Face Recognition Library.
This library is designed to perform a fair comparison of face recognition algorithms.
It contains scripts to execute various kinds of face recognition experiments on a variety of facial image databases.

Installation
------------

Currently the library is only available as a private project of `bioidiap at GitHub`_.
To check out the current version of the FaceRecLib, go to the console, move to any place you like and call::

  $ git clone git@github.com:bioidiap/facereclib.git

.. note::
  If you already have an older version of the FaceRecLib, just re-base the repository by calling (from within your local copy)::

    $ git config remote.origin.url git@github.com:bioidiap/facereclib.git
    $ git pull


The FaceRecLib is a satellite package of Bob_.
You will need a copy of it to run the algoritms.
Please download Bob_ from its webpage.

.. note::
  At Idiap_, Bob_ is globally installed and you can use its current version.

After downloading, you should go to the console and write::

  $ python bootstrap.py
  $ bin/buildout
  $ bin/sphinx

.. note::
  Here at Idiap_, please replace the first command by::

    $ /idiap/group/torch5spro/externals/v3/ubuntu-10.04-x86_64/bin/python bootstrap.py


Now, you can open the documentation by typing::

  $ firefox sphinx/html/index.html

and read further instructions on how to use this library.

.. _bob: http://www.idiap.ch/software/bob
.. _idiap: http://www.idiap.ch
.. _bioidiap at github: http://www.github.com/bioidiap
