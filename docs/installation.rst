.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

=========================
Installation instructions
=========================

Download
--------

|project|
~~~~~~~~~
Currently the library is only available as a private project of `Idiap at GitHub`_.
To check out the current version of the |project|, go to the console, move to any place you like and call::

.. code-block:: sh

  $ git clone git@github.com:bioidiap/facereclib.git

Bob
~~~

The |project| is a satellite package of Bob_, where most of the image processing, feature extraction, and face recognition algorithms, as well as the evaluation techniques are implemented.
To run properly, |project| requires Bob_ at least in version 1.1.0.
If you have not installed Bob_ or your version is to old, please visit `Bob's GitHub page`_ and download the latest version.
If you prefer not to install Bob_ in its default location, you have to adapt the ``eggs-directories`` of the **buildout.cfg** configuration file, which sits in the main directory of the |project|.

.. note::
  Currently, the version 1.1.0 of Bob is not yet released.
  Hence, if you are not at Idiap_ (see **Using |project| at Idiap** below) you have bad luck.

.. note::
  Currently, there is no MS Windows version of Bob_.

Using Bob at Idiap
~~~~~~~~~~~~~~~~~~

At Idiap, you can use the latest version of Bob_ by changing the ``eggs-directories`` to **/idiap/group/torch5spro/nightlies/last/install/linux-x86_64-release/lib** (the current default).
But if you prefer to use the latest stable version, just comment out the ``eggs-directories``.

.. note::
  Currently, the latest stable version of Bob_ is not compatible with the |project|.

Of course, you can also use your own private copy of Bob_, just set the right ``eggs-directories``.

Usually, the database satellite packages (called ``xbob.db.[...]``) are automatically downloaded from PyPI_.
If you prefer to use the latest version of them, just add ``find-links = http://www.idiap.ch/software/bob/packages/xbob/nightlies/last`` into your **buildout.cfg** file.

.. note::
  Currently, the database packages are not yet uploaded to PyPI_.
  Hence, you **have to** use the latest versions.

Image databases
~~~~~~~~~~~~~~~

With the |project| you will run face recognition experiments using some default facial image databases.
Though the verification protocols are implemented in the |project|, the images (and the hand-labelled annotations) are **not included**.
To download the image databases, please refer to the according Web-pages.
For a start, you might want to try the small, but freely available image database called the `AT&T database`_ (formerly known as the ORL database).


Set-up your |project|
---------------------

Now, you have everything set up such that you can continue to set up the |project|.
To do this, we use the BuildOut_ system.
To proceed, open a terminal in your |project| main directory and call:

.. code-block:: sh

  $ python bootstrap.py
  $ bin/buildout

The first step will generate a **bin** directory in the main directory of the |project|.
The second step automatically downloads all dependencies (except for Bob_) of the |project| and creates all required scripts that we will need soon.

.. note::
  At Idiap_, you should not use the default python interpreter to call the bootstrapping.
  To get all the dependencies of Bob_ right, you should instead use:

  .. code-block:: sh

    $ /idiap/group/torch5spro/externals/v3/ubuntu-10.04-x86_64/bin/python bootstrap.py


Test your installation
~~~~~~~~~~~~~~~~~~~~~~

One of the scripts that were generated during the bootstrap/buildout step is a test script.
To verify your installation, you should run the script by calling:

.. code-block:: sh

  $ bin/tests

.. note::
  In case any of the tests fail, please file a bug through the `GitHub bug system`_.

Generate this documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To generate this documentation, you call:

.. code-block:: sh

  $ bin/sphinx

Afterwards, the documentation is available and you can read it, e.g., by using:

.. code-block:: sh

  $ firefox sphinx/html/index.html

(Since you are reading the documentation, you already have done this step, right?)


.. include:: links.rst
