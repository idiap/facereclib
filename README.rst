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

- Eigenfaces [M. Turk and A. Pentland. "Eigenfaces for recognition". Journal of Cognitive Neuroscience, 3(1):71-86, 1991.]
- Linear Discriminant Analysis [W. Zhao, A. Krishnaswamy, R. Chellappa, D. Swets and J. Weng. "Discriminant analysis of principal components for face recognition", pages 73-85. Springer Verlag Berlin, 1998.]
- Probabilistic Linear Discriminant Analysis [L. El Shafey, C. McCool, R. Wallace and S. Marcel. "A scalable formulation of probabilistic linear discriminant analysis: applied to face recognition". IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(7):1788-1794, 7/2013.]
- Local Gabor Binary Pattern Histogram Sequences [W. Zhang, S. Shan, W. Gao, X. Chen and H. Zhang. "Local Gabor binary pattern histogram sequence (LGBPHS): a novel non-statistical model for face representation and recognition". Computer Vision, IEEE International Conference on, 1:786-791, 2005.]
- Graph Matching [M. Günther, D. Haufe and R.P. Würtz. "Face recognition with disparity corrected Gabor phase differences". In Artificial neural networks and machine learning, volume 7552 of Lecture Notes in Computer Science, pages 411-418. 2012.]
- Gaussian Mixture Modeling [C. McCool, S. Marcel. "Parts-based face verification using local frequency bands". In Advances in biometrics, volume 5558 of Lecture Notes in Computer Science. 2009.]
- Inter-Session Variability Modeling [R. Wallace, M. McLaren, C. McCool and S. Marcel. "Inter-session variability modelling and joint factor analysis for face authentication". International Joint Conference on Biometrics. 2011.]
- Bayesian Intrapersonal/Extrapersonal Classifier [B. Moghaddam, W. Wahid and A. Pentland. "Beyond eigenfaces: probabilistic matching for face recognition". IEEE International Conference on Automatic Face and Gesture Recognition, pages 30-35. 1998.]

is provided.
Furthermore, tools to evaluate the results can easily be used to create scientific plots, and interfaces to run experiments using parallel processes or an SGE grid are provided.

Extensions
..........
On top of these already pre-coded algorithms, the FaceRecLib provides an easy Python interface for implementing new image preprocessors, feature types, face recognition algorithms or database interfaces, which directly integrate into the face recognition experiment.
Hence, after a short period of coding, researchers can compare their new invention directly with already existing algorithms in a fair manner.

As a small example, we provide wrapper classes for the CSU face recognition resources [http://www.cs.colostate.edu/facerec] in the `xfacereclib.CSU.PythonFaceEvaluation <http://pypi.python.org/pypi/xfacereclib.CSU.PathonFaceEvaluation>`_ package.
To see how easy it is to use these tools to generate a publication, please have a look at the source code `xfacereclib.paper.BeFIT2012 <http://pypi.python.org/pypi/xfacereclib.paper.BeFIT2012>`_, which was finally published in [M. Günther, R. Wallace and S. Marcel. "An Open Source Framework for Standardized Comparisons of Face Recognition Algorithms". Computer Vision - ECCV 2012. Workshops and Demonstrations, LNCS, 7585, 547-556, 2012.].

Finally, all parts of the FaceRecLib are well documented and thoroughly tested to assure usability, stability and comparability.

Interested?
...........
If you are interested in trying out the FaceRecLib, please follow the installation instruction below.


Installation
------------

We proudly present the first version of the FaceRecLib on pypi.
To download the FaceRecLib, please go to http://pypi.python.org/pypi/facereclib, click on the download button and extract the .zip file to a folder of your choice.

Bob
...

The FaceRecLib is a satellite package of the free signal processing and machine learning library Bob_.
You will need a copy of Bob in version 1.2.0 it to run the algorithms.
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
