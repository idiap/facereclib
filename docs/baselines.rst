.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

Executing baseline algorithms
=============================

The first thing you might want to do is to execute one of the baseline algorithms that are implemented in the |project|.

Setting up your database
------------------------

As already mentioned, the image databases are not included in this package, so you have to download them.
For example, you can easily download the images of the `AT&T database`_.
After extracting the images, you have to modify the according configuration file, e.g. **config/database/atnt_Default.py** by correcting the ``image_directory`` entry.

.. note::
  The directories in the configuration files are preset to the right directories at Idiap_.
  Hence, at Idiap_ you don't need to care about that.


Structure of an experiment in the |project|
-------------------------------------------

Each experiment is divided into several steps.
The steps are:

1. Image preprocessing: Raw images are aligned and photometrically enhanced.
2. Feature extractor training: The feature extraction parameters are learned.
3. Feature extraction: The features are extracted from the preprocessed images.
4. Feature projector training: Parameters of a subspace-projection of the features are learned.
5. Feature projection: The extracted features are projected into a subspace.
6. Model enroller training: The ways how to extract models is learned.
7. Model enrollment: One model is enrolled from the features of one or more images.
8. Scoring: The verification scores between various models and probe features are computed.
9. Evaluation: The computed scores are evaluated.

The communication between two steps is file-based.
The output of one step usually serves as the input of the subsequent step(s).
Depending on the algorithm, some of the steps is not applicable/available.
E.g. most of the feature extractors do not need a special training step, or some algorithms do not require a subspace projection.
In these cases, the according steps are skipped.
The |project| takes care that always the correct files are forwarded to the subsequent steps.


Running experiments
-------------------

To run the baseline experiments, you can use the ``bin/baselines.py`` script.
This script is a simple wrapper for the ``bin/faceverify.py`` script that will be explained in more detail in the :doc:experiments section.
The ``--help`` option shows you, which other options you have.
Here is an almost complete extract:

* ``--database``: The image database you want to use.
  By default this is set to atnt.
* ``--protocol``: The evaluation protocol, which is specific for the database.
  By default this is set to the Default protocol of the AT&T database.
* ``--algorithms``: The face recognition algorithms that you want to execute.
  By default, only the Eigenface algorithm is executed.
* ``--all``: Execute all algorithms that are implemented.
* ``--directory``: The directory where the files of the experiments are put to.
  If not specified, by default the files are split up into the temporary files and the result files, see the ``--temp-directory`` and the ``--user-directory`` of ``bin/faceverify.py --help``.
  In this script, if the ``--directory`` option is specified, all files will be put into the given directory.
* ``--evaluate``: After running the experiments, the resulting score files will be evaluated, and the result is written to console.
* ``--dry-run``: Instead of executing the algorithm (or the evaluation), only print the command that would have been executed.
* ``--verbose``: Increase the verbosity level of the script.
  By default, only the commands that are executed are printed, and the rest of the calculation runs quietly.
  You can increase the verbosity by adding the ``--verbose`` parameter repeatedly (up to three times).

Usually it is a good idea to have at least verbose level 2 (i.e., calling ``bin/baselines.py --verbose --verbose``, or the short version ``bin/baselines.py -vv``).

.. note::
  The directories are set up such that you can use them at Idiap_ without further modifications.


The algorithms
--------------

The algorithms present an (incomplete) set of state-of-the-art face recognition algorithms. Here is the list of short-cuts:

* ``eigenface``: The eigenface algorithm as proposed by [TP91]_. It uses the pixels as raw data, and applies a *Principal Component Analysis* (PCA) on it.

* ``lda``: The LDA algorithm applies a *Linear Discriminant Analysis* (LDA), here we use the combined PCA+LDA approach [ZKC+98]_ .

* ``gaborgraph``: This method extract grid graphs of Gabor jets from the images, and computes a Gabor phase based similarity [GHW12]_.

* ``lgbphs``: *Local Gabor Binary Pattern Histogram Sequences* (LGBPHS) [ZSG+05]_ are extracted from the images and compares using the histogram intersection measure.

* ``gmm``: *Gaussian Mixture Models* (GMM) [WMM+12]_ are extracted from *Discrete Cosine Transform* (DCT) block features.

* ``isv``: As an extension of the GMM algorithm, *Inter-Session Variability* (ISV) modelling [WMM+11]_ is used to learn what variations in images are introduced by identity changes and which not.

* ``plda``: *Probabilistic LDA* (PLDA) [Pri07]_ is a probabilistic generative version of the LDA.
  Here, we also apply it on pixel-based representations of the image, though also other features should be possible.

* ``bic``: In the *Bayesian Intrapersonal/Extrapersonal Classifier* (BIC) [MWP98]_, a pixel-based difference image is classified to be intrapersonal (i.e., both images are from the same person) or extrapersonal.

.. [TP91]    M. Turk and A. Pentland. Eigenfaces for recognition. Journal of Cognitive Neuroscience, 3(1):71-86, 1991.
.. [ZKC+98]  W. Zhao, A. Krishnaswamy, R. Chellappa, D. Swets and J. Weng. Discriminant analysis of principal components for face recognition, pages 73-85. Springer Verlag Berlin, 1998.
.. [GHW12]   M. Günther, D. Haufe and R.P. Würtz. Face recognition with disparity corrected Gabor phase differences. International Conference on Artificial Neural Networks, pages .... 2012.
.. TODO:: Correct the citation of this paper.
.. [ZSG+05]  W. Zhang, S. Shan, W. Gao, X. Chen and H. Zhang. Local Gabor binary pattern histogram sequence (LGBPHS): a novel non-statistical model for face representation and recognition. Computer Vision, IEEE International Conference on, 1:786-791, 2005.
.. [WMM+12]  R. Wallace, M. McLaren, C. McCool and S. Marcel. Cross-pollination of normalisation techniques from speaker to face authentication using Gaussian mixture models. IEEE Transactions on Information Forensics and Security, 2012.
.. TODO:: Is this the right citation?
.. [WMM+11]  R. Wallace, M. McLaren, C. McCool and S. Marcel. Inter-session variability modelling and joint factor analysis for face authentication. International Joint Conference on Biometrics. 2011.
.. [Pri07]   S. J. D. Prince. Probabilistic linear discriminant analysis for inferences about identity. Proceedings of the International Conference on Computer Vision. 2007.
.. [MWP98]   B., Moghaddam, W. Wahid and A. Pentland. Beyond eigenfaces: probabilistic matching for face recognition. IEEE International Conference on Automatic Face and Gesture Recognition, pages 30-35. 1998.


Baseline results
----------------

The results of the baseline experiments are generated using the ``--evaluate`` option of the ``bin/baselines.py`` script.
For the `AT&T database`_ the results should be as follows:

.. table:: The HTER results of the baseline algorithms on the AT&T database

  +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
  |  eigenface  |     lda     |  gaborgraph |    lgbphs   |     gmm     |     isv     |    plda     |     bic     |
  +=============+=============+=============+=============+=============+=============+=============+=============+
  |   30.842%   |   33.079%   |    7.000%   |   10.000%   |    1.000%   |    0.053%   |   44.000%   |   47.895%   |
  +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+

.. note::
  Here, only the results of the HTER using the EER as minimum criterion is given.

.. note::
  ``bin/baselines.py --evaluate`` prints results of the development and the test set.
  For the AT&T database, there is actually no test set.
  Hence, the result of the development set is printed twice.

.. include:: links.rst
