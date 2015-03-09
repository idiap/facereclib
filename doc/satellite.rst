.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Fri Oct 26 17:05:40 CEST 2012

.. _satellite-packages:

=================================
Create your own Satellite Package
=================================

The simplest and most clean way to test your own code in the FaceRecLib is to add it in a satellite package.
In principle, you can choose a name that fits you, but it is preferable to have a package name that starts with ``xfacereclib.`` to show that it is a satellite package to the FaceRecLib.
Please refer to the :ref:`satellite package explanation of Bob <bob.extension>`, which explains in detail how to start.

Depending on of what nature is your contribution, you have to register it in the `setup.py` file of your satellite package.
In case, your contribution is a face recognition algorithm, you might want to :ref:`register it <register-resources>`.
After doing that, you can simply use the ``./bin/faceverify.py`` (or any other script of the FaceRecLib) with your registered tool, as if it would be part of the FaceRecLib.
As one example of providing a source code package, you might want to have a look into the wrapper classes xfacereclib.extension.CSU_ for the `CSU Face Recognition Resources`_.

Another contribution of code is to provide the source code to rerun the experiments as published in a paper.
In this case, the contribution is more about scripts that can be used to run experiments.
To cause the buildout_ system to create a python script in the `bin <file:../bin>`_ directory, you have to register the script in your `setup.py` file under the ``console_scripts`` section.
One working example of providing source code to rerun experiments for [GWM12]_ can be found in http://pypi.python.org/pypi/xfacereclib.paper.BeFIT2012.


Contribute your Code
--------------------
When you invented a completely new type of preprocessing, features, or recognition algorithm and you want to share them with the world, or you want other researchers to be able to rerun your experiments, you are highly welcome **and encouraged** to do so.
Please make sure that every part of your code is documented and tested.

To upload your satellite package to the world (more specifically to PyPI_) you have to create an account and register an ssh key.
Add the required packages in the `setup.py` file and under the ``install_requires`` section, provide the other information and upload the package to PyPI via:

.. code-block:: sh

  $ python setup.py register
  $ python setup.py sdist --formats zip upload

Now, all other researchers can make use of your invention, with the effect that your paper will be cited more often, simply by adding your project to the **setup.py** in their satellite package.

.. include:: links.rst
