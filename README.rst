.. vim: set fileencoding=utf-8 :
.. Manuel Guenther <manuel.guenther@idiap.ch>
.. Fri Sep 19 12:51:09 CEST 2014

.. image:: http://img.shields.io/badge/docs-stable-yellow.png
   :target: http://pythonhosted.org/facereclib/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.png
   :target: https://www.idiap.ch/software/bob/docs/latest/idiap/facereclib/master/index.html
.. image:: https://travis-ci.org/idiap/facereclib.svg?branch=v2.1.1
   :target: https://travis-ci.org/idiap/facereclib?branch=v2.1.1
.. image:: https://coveralls.io/repos/idiap/facereclib/badge.png?branch=v2.1.1
   :target: https://coveralls.io/r/idiap/facereclib?branch=v2.1.1
.. image:: https://img.shields.io/badge/github-master-0000c0.png
   :target: https://github.com/idiap/facereclib/tree/master
.. image:: http://img.shields.io/pypi/v/facereclib.png
   :target: https://pypi.python.org/pypi/facereclib
.. image:: http://img.shields.io/pypi/dm/facereclib.png
   :target: https://pypi.python.org/pypi/facereclib


==============================
 The Face Recognition Library
==============================

Welcome to the Face Recognition Library.
This library is designed to perform a fair comparison of face recognition algorithms.
It contains scripts to execute various kinds of face recognition experiments on a variety of facial image databases, and running baseline algorithms is as easy as going to the command line and typing::

  $ bin/baselines.py --database frgc --algorithm eigenface


About
-----

This library is developed at the `Biometrics group <http://www.idiap.ch/~marcel/professional/Research_Team.html>`_ at the `Idiap Research Institute <http://www.idiap.ch>`_.
The FaceRecLib is designed to run face recognition experiments in a comparable and reproducible manner.

.. note::
  When you are working at Idiap_, you might get a version of the FaceRecLib, where all paths are set up such that you can directly start running experiments.
  Outside Idiap_, you need to set up the paths to point to your databases, please check `Read Further`_ on how to do that.

Databases
.........
To achieve this goal, interfaces to many publicly available facial image databases are contained, and default evaluation protocols are defined, e.g.:

- Face Recognition Grand Challenge version 2 [http://www.nist.gov/itl/iad/ig/frgc.cfm]
- The Good, The Bad and the Ugly [http://www.nist.gov/itl/iad/ig/focs.cfm]
- Labeled Faces in the Wild [http://vis-www.cs.umass.edu/lfw]
- Multi-PIE [http://www.multipie.org]
- SCface [http://www.scface.org]
- MOBIO  [http://www.idiap.ch/dataset/mobio]
- BANCA [http://www.ee.surrey.ac.uk/CVSSP/banca]
- CAS-PEAL [http://www.jdl.ac.cn/peal/index.html]
- AR face database [http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html]
- XM2VTS [http://www.ee.surrey.ac.uk/CVSSP/xm2vtsdb]
- The AT&T database of faces (formerly known as ORL) [http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html]

Algorithms
..........
Together with that, a broad variety of traditional and state-of-the-art face recognition algorithms such as:

- Eigenfaces [TP91]_
- Linear Discriminant Analysis [ZKC+98]_
- Probabilistic Linear Discriminant Analysis [ESM+13]_
- Local Gabor Binary Pattern Histogram Sequences [ZSG+05]_
- Graph Matching [GHW12]_
- Gaussian Mixture Modeling [MM09]_
- Inter-Session Variability Modeling [WMM+11]_
- Bayesian Intrapersonal/Extrapersonal Classifier [MWP98]_

is provided.
Furthermore, tools to evaluate the results can easily be used to create scientific plots, and interfaces to run experiments using parallel processes or an SGE grid are provided.

Extensions
..........
On top of these already pre-coded algorithms, the FaceRecLib provides an easy Python interface for implementing new image preprocessors, feature types, face recognition algorithms or database interfaces, which directly integrate into the face recognition experiment.
Hence, after a short period of coding, researchers can compare their new invention directly with already existing algorithms in a fair manner.
Some of this code is provided as extensions of the FaceRecLib.

The CSU Face Recognition Resources
++++++++++++++++++++++++++++++++++

As a small example, we provide wrapper classes for the CSU face recognition resources, which you can download `here <http://www.cs.colostate.edu/facerec>`__ in the xfacereclib.extension.CSU_ package.
We have implemented wrappers for two face recognition algorithms:

- Local Region PCA [PBD+11]_
- LDA-IR (a.k.a. CohortLDA) [LBP+12]_

To see how easy it is to use these tools to generate a publication, please have a look at the source code `xfacereclib.paper.BeFIT2012 <http://pypi.python.org/pypi/xfacereclib.paper.BeFIT2012>`_, with the according `Paper <http://publications.idiap.ch/index.php/publications/show/2431>`_ being published in [GWC12]_.

Finally, all parts of the FaceRecLib are well documented and thoroughly tested to assure usability, stability and comparability.

References
..........

.. [TP91]    *M. Turk and A. Pentland*. **Eigenfaces for recognition**. Journal of Cognitive Neuroscience, 3(1):71-86, 1991.
.. [ZKC+98]  *W. Zhao, A. Krishnaswamy, R. Chellappa, D. Swets and J. Weng*. **Discriminant analysis of principal components for face recognition**, pages 73-85. Springer Verlag Berlin, 1998.
.. [ESM+13]  *L. El Shafey, C. McCool, R. Wallace and S. Marcel*. **A scalable formulation of probabilistic linear discriminant analysis: applied to face recognition**. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(7):1788-1794, 7/2013.
.. [ZSG+05]  *W. Zhang, S. Shan, W. Gao, X. Chen and H. Zhang*. **Local Gabor binary pattern histogram sequence (LGBPHS): a novel non-statistical model for face representation and recognition**. Computer Vision, IEEE International Conference on, 1:786-791, 2005.
.. [GHW12]   *M. Guenther, D. Haufe and R.P. Wuertz*. **Face recognition with disparity corrected Gabor phase differences**. In Artificial neural networks and machine learning, volume 7552 of Lecture Notes in Computer Science, pages 411-418. 9/2012.
.. [MM09]    *C. McCool, S. Marcel*. **Parts-based face verification using local frequency bands**. In Advances in biometrics, volume 5558 of Lecture Notes in Computer Science. 2009.
.. [WMM+11]  *R. Wallace, M. McLaren, C. McCool and S. Marcel*. **Inter-session variability modelling and joint factor analysis for face authentication**. International Joint Conference on Biometrics. 2011.
.. [MWP98]   *B. Moghaddam, W. Wahid and A. Pentland*. **Beyond eigenfaces: probabilistic matching for face recognition**. IEEE International Conference on Automatic Face and Gesture Recognition, pages 30-35. 1998.
.. [PBD+11]  *P.J. Phillips, J.R. Beveridge, B.A. Draper, G. Givens, A.J. O'Toole, D.S. Bolme, J. Dunlop, Y.M. Lui, H. Sahibzada and S. Weimer*. **An introduction to the Good, the Bad, & the Ugly face recognition challenge problem**. Automatic Face Gesture Recognition and Workshops (FG 2011), pages 346-353. 2011.
.. [LBP+12]  *Y.M. Lui, D.S. Bolme, P.J. Phillips, J.R. Beveridge and B.A. Draper*. **Preliminary studies on the Good, the Bad, and the Ugly face recognition challenge problem**. Computer Vision and Pattern Recognition Workshops (CVPRW), pages 9-16. 2012.
.. [GWC12]   *M. Guenther, R. Wallace and S. Marcel*. **An Open Source Framework for Standardized Comparisons of Face Recognition Algorithms**. Computer Vision - ECCV 2012. Workshops and Demonstrations, LNCS, 7585, 547-556, 2012.


Installation
------------

To download the FaceRecLib, please go to http://pypi.python.org/pypi/facereclib, click on the **download** button and extract the .zip file to a folder of your choice.

The FaceRecLib is a satellite package of the free signal processing and machine learning library Bob_, and some of its algorithms rely on the `CSU Face Recognition Resources`_.
These two dependencies have to be downloaded manually, as explained in the following.


Bob
...

This version of the FaceRecLib relies on Bob_ version 2 or greater.
To install `Packages of Bob <https://github.com/idiap/bob/wiki/Packages>`_, please read the `Installation Instructions <https://github.com/idiap/bob/wiki/Installation>`_.
For Bob_ to be able to work properly, some dependent packages are required to be installed.
Please make sure that you have read the `Dependencies <https://github.com/idiap/bob/wiki/Dependencies>`_ for your operating system.

The most simple solution is to download and extract this package, go to the console and write::

  $ python bootstrap.py
  $ bin/buildout

This will download all required dependent packages and install them locally.
If you don't want all the database packages to be downloaded, please remove the bob.db.[database] lines from the ``eggs`` section of the file **buildout.cfg** in the main directory before calling the three commands above.


The CSU Face Recognition Resources
..................................

Two open source algorithms are provided by the `CSU Face Recognition Resources`_, namely the LRPCA and the LDA-IR (a.k.a. CohortLDA) algorithm.
For these algorithms, optional wrapper classes are provided in the xfacereclib.extension.CSU_ satellite package.
By default, this package is disabled.
To enable the two algorithms, please call::

  $ bin/buildout -c buildout-with-csu.cfg

after downloading and patching the CSU resources, and updating the ``csu-dir`` in the **buildout-with-csu.cfg** file -- as explained in xfacereclib.extension.CSU_.


Test your installation
......................

To verify that your installation worked as expected, you might want to run our test utilities::

  $ bin/nosetests

Usually, all tests should pass, if you use the latest packages of Bob_.
With other versions of Bob_, you might find some failing tests, or some errors might occur.


Read further
------------

For further documentation on this package, please read the `Stable Version <http://pythonhosted.org/facereclib/index.html>`_ or the `Latest Version <https://www.idiap.ch/software/bob/docs/latest/idiap/facereclib/master/index.html>`_ of the documentation.
For a list of tutorials on this or the other packages ob Bob_, or information on submitting issues, asking questions and starting discussions, please visit its website.

Generate a local documentation
..............................

There are several file links in the documentation, which won't work correctly in the online documentation.
To generate the documentation locally, type::

  $ bin/sphinx-build doc sphinx
  $ firefox sphinx/index.html

and read further instructions on how to use this library.

.. note::
  Some links in the documentation require that the documentation is generated exactly with the command line ``bin/sphinx-build doc sphinx`` (see above).
  If you generated the documentation using another command line, please be aware that file links might not be found either.


Cite our paper
--------------

If you use the FaceRecLib in any of your experiments, please cite the following paper::

  @inproceedings{Guenther_BeFIT2012,
         author = {G{\"u}nther, Manuel AND Wallace, Roy AND Marcel, S{\'e}bastien},
         editor = {Fusiello, Andrea AND Murino, Vittorio AND Cucchiara, Rita},
       keywords = {Biometrics, Face Recognition, Open Source, Reproducible Research},
          month = oct,
          title = {An Open Source Framework for Standardized Comparisons of Face Recognition Algorithms},
      booktitle = {Computer Vision - ECCV 2012. Workshops and Demonstrations},
         series = {Lecture Notes in Computer Science},
         volume = {7585},
           year = {2012},
          pages = {547-556},
      publisher = {Springer Berlin},
       location = {Heidelberg},
            url = {http://publications.idiap.ch/downloads/papers/2012/Gunther_BEFIT2012_2012.pdf}
  }


Trouble shooting
----------------

In case of further question or problems, feel free to start a new discussion in our `Mailing List <https://groups.google.com/forum/#!forum/bob-devel>`_.
When you have identified a clear bug, please go ahead and open a new ticket in our `Issue Tracker <http://www.github.com/bioidiap/facereclib/issues>`_.


.. _bob: http://www.idiap.ch/software/bob
.. _idiap: http://www.idiap.ch
.. _bioidiap at github: http://www.github.com/bioidiap
.. _csu face recognition resources: http://www.cs.colostate.edu/facerec
.. _xfacereclib.extension.csu: http://pypi.python.org/pypi/xfacereclib.extension.CSU
