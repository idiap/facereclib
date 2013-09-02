The Face Recognition Library
============================

Welcome to the Face Recognition Library.
This library is designed to perform a fair comparison of face recognition algorithms.
It contains scripts to execute various kinds of face recognition experiments on a variety of facial image databases, and running baseline algorithms is as easy as going to the command line and typing::

  $ bin/baselines.py --database frgc --algorithm lda


About
-----

This library is developed at the `Biometrics group <http://www.idiap.ch/~marcel/professional/Research_Team.html>`_ at the `Idiap Research Institute <http://www.idiap.ch>`_.
The FaceRecLib is designed to run face recognition experiments in a comparable and reproducible manner.

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

As a small example, we provide wrapper classes for the CSU face recognition resources [http://www.cs.colostate.edu/facerec] in the `xfacereclib.extension.CSU <http://pypi.python.org/pypi/xfacereclib.extension.CSU>`_ package:

- Local Region PCA [PBD+11]_
- LDA-IR (a.k.a. CohortLDA) [LBP+12]_


To see how easy it is to use these tools to generate a publication, please have a look at the source code `xfacereclib.paper.BeFIT2012 <http://pypi.python.org/pypi/xfacereclib.paper.BeFIT2012>`_, which was finally published in [M. Günther, R. Wallace and S. Marcel. "An Open Source Framework for Standardized Comparisons of Face Recognition Algorithms". Computer Vision - ECCV 2012. Workshops and Demonstrations, LNCS, 7585, 547-556, 2012.].

Finally, all parts of the FaceRecLib are well documented and thoroughly tested to assure usability, stability and comparability.

References
..........

.. [TP91]    *M. Turk and A. Pentland*. **Eigenfaces for recognition**. Journal of Cognitive Neuroscience, 3(1):71-86, 1991.
.. [ZKC+98]  *W. Zhao, A. Krishnaswamy, R. Chellappa, D. Swets and J. Weng*. **Discriminant analysis of principal components for face recognition**, pages 73-85. Springer Verlag Berlin, 1998.
.. [ESM+13]  *L. El Shafey, Chris McCool, Roy Wallace and Sébastien Marcel*. **A scalable formulation of probabilistic linear discriminant analysis: applied to face recognition**. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(7):1788-1794, 7/2013.
.. [ZSG+05]  *W. Zhang, S. Shan, W. Gao, X. Chen and H. Zhang*. **Local Gabor binary pattern histogram sequence (LGBPHS): a novel non-statistical model for face representation and recognition**. Computer Vision, IEEE International Conference on, 1:786-791, 2005.
.. [GHW12]   *M. Günther, D. Haufe and R.P. Würtz*. **Face recognition with disparity corrected Gabor phase differences**. In Artificial neural networks and machine learning, volume 7552 of Lecture Notes in Computer Science, pages 411-418. 9/2012.
.. [MM09]    *C. McCool, S. Marcel*. **Parts-based face verification using local frequency bands**. In Advances in biometrics, volume 5558 of Lecture Notes in Computer Science. 2009.
.. [WMM+11]  *R. Wallace, M. McLaren, C. McCool and S. Marcel*. **Inter-session variability modelling and joint factor analysis for face authentication**. International Joint Conference on Biometrics. 2011.
.. [MWP98]   *B. Moghaddam, W. Wahid and A. Pentland*. **Beyond eigenfaces: probabilistic matching for face recognition**. IEEE International Conference on Automatic Face and Gesture Recognition, pages 30-35. 1998.
.. [PBD+11]  *P.J. Phillips, J.R. Beveridge, B.A. Draper, G. Givens, A.J. O'Toole, D.S. Bolme, J. Dunlop, Y.M. Lui, H. Sahibzada and S. Weimer*. **An introduction to the Good, the Bad, & the Ugly face recognition challenge problem**. Automatic Face Gesture Recognition and Workshops (FG 2011), pages 346-353. 2011.
.. [LBP+12]  *Y.M. Lui, D.S. Bolme, P.J. Phillips, J.R. Beveridge and B.A. Draper*. **Preliminary studies on the Good, the Bad, and the Ugly face recognition challenge problem**. Computer Vision and Pattern Recognition Workshops (CVPRW), pages 9-16. 2012.


Installation
------------

We proudly present the first version of the FaceRecLib on pypi.
To download the FaceRecLib, please go to http://pypi.python.org/pypi/facereclib, click on the download button and extract the .zip file to a folder of your choice.

The FaceRecLib is a satellite package of the free signal processing and machine learning library Bob_, and some of its algorithms rely on the `CSU Face Recognition Resources`_.
These two dependencies have to be downloaded manually, as explained in the following.

Bob
...

You will need a copy of Bob in version 1.2.0 to run the algorithms.
Please download Bob_ from its webpage.

.. note::
  At Idiap_, Bob_ is globally installed.
  This version of the FaceRecLib is bound to Bob version 1.2.0, which does not correspond to the one installed.
  However, the correct version of Bob is marked in the buildout.cfg.

After downloading, you should go to the console and write::

  $ python bootstrap.py
  $ bin/buildout
  $ bin/sphinx-build docs sphinx

This will download all required packages and install them locally.
If you don't want all the database packages to be downloaded, please remove the xbob.db.[database] lines from the ``eggs`` section of the file **buildout.cfg** in the main directory before calling the three commands above.

Now, you can open the documentation by typing::

  $ firefox sphinx/index.html

and read further instructions on how to use this library.

The CSU Face Recognition Resources
..................................
Two open source algorithms are provided by the `CSU Face Recognition Resources`_, namely the LRPCA and the LDA-IR (a.k.a. CohortLDA) algorithm.
For these algorithms, optional wrapper classes are provided in the xfacereclib.extension.CSU_ satellite package.
By default, this package is disabled.
To enable them, please call::

  $ bin/buildout -c buildout-with-csu.cfg

after downloading and patching the CSU resources, and updating the ``sources-dir`` in the **buildout-with-csu.cfg** file -- as explained in xfacereclib.extension.CSU_.


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


.. _bob: http://www.idiap.ch/software/bob
.. _idiap: http://www.idiap.ch
.. _bioidiap at github: http://www.github.com/bioidiap
.. _csu face recognition resources: http://www.cs.colostate.edu/facerec
.. _xfacereclib.extension.csu: http://pypi.python.org/pypi/xfacereclib.extension.CSU


