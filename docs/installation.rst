.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

Installation instructions
=========================

Download
--------

|project|
~~~~~~~~~
Currently the library is only available as a private project of `Idiap at GitHub`_.
To check out the current version of the |project|, go to the console, move to any place you like and call::

  $ git clone git@github.com:bioidiap/facereclib.git

Bob
~~~

The |project| is a satellite package of `Bob`_, where most of the image processing, feature extraction, and face recognition algorithms, as well as the evaluation techniques are implemented.
To run properly, |project| requires `Bob`_ at least in version 1.1.0.
If you have not installed `Bob`_ or your version is to old, please visit `Bob's GitHub page`_ and download the latest version.
If you prefer not to install `Bob`_ in its default location, you have to adapt the ``eggs-directories`` of the **buildout.cfg** configuration file, which sits in the main directory of the |project|

.. note::

  Currently, the version 1.1.0 of Bob is not yet released.
  Hence, if you are not at `Idiap`_ (see **Using |project| at Idiap** below) you have bad luck.

.. note::

  Currently, there is no MS Windows version of `Bob`_.

Using Bob at Idiap
~~~~~~~~~~~~~~~~~~

At Idiap, you can use the latest version of `Bob`_ by changing the ``eggs-directories`` to **/idiap/group/torch5spro/nightlies/last/install/linux-x86_64-release/lib** (the current default).
But if you prefer to use the latest stable version, just comment out the ``eggs-directories``.

.. note::

  Currently, the latest stable version of `Bob`_ is not compatible with the |project|.

Of course, you can also use your own private copy of `Bob`_, just set the right ``eggs-directories``.

Image databases
~~~~~~~~~~~~~~~

With the |project| you will run face recognition experiments using some default facial image databases. Though the verification protocols are implemented in the |project|, the images (and the hand-labeled annotations) are **not included**. To download the image databases, please refer to the according Webpages. For a start, one small, but freely available image database is the `AT&T database`_ (formerly known as the ORL database).


Setup
-----

Now, you have everything set up such that you can continue to set up the |project|.
To do this, we use the `BuildOut`_ system.
To proceed, open a terminal in your |project| main directory and call:

.. code-block:: sh

  $ python bootstrap.py
  $ bin/buildout

The first step will generate a **bin** directory in the main directory of the |project|.
The second step automatically downloads all dependencies (except for `Bob`_) of the |project| and creates all required scripts that we will need soon.

.. note::

  At `Idiap`_, you should not use the default python interpreter to call the bootstapping.
  To get all the dependencies of `Bob`_ right, you should instead use:

  .. code-block:: sh

    $ /remote/filer.gx/group.torch5spro/nightlies/externals/v3/ubuntu-10.04-x86_64/bin/python bootstrap.py


Test your installation
~~~~~~~~~~~~~~~~~~~~~~

One of the scripts that were generated during the bootstrap/buildout step is a test script.
To verify your installation, you should run the script by calling:

.. code-block:: sh

  $ bin/tests

Generate this documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To generate this documentation, you call:

.. code-block:: sh

  $ bin/sphinx

Afterwards, the documentation is available and you can read it, e.g., by using:

.. code-block:: sh

  $ firefox sphinx/html/index.html

(But you already have done this step, right?)


.. include:: links.rst
