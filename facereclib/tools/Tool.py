#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Tue Oct  2 12:12:39 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy
import os
from .. import utils

class Tool:
  """This is the base class for all face recognition tools.
  It defines the minimum requirements for all derived tool classes.
  """

  def __init__(
      self,
      performs_projection = False, # enable if your tool will project the features
      requires_projector_training = True, # by default, the projector needs training, if projection is enabled
      split_training_features_by_client = False, # enable if your projector training needs the training files sorted by client
      use_projected_features_for_enrollment = True, # by default, the enroller used projected features for enrollment, if projection is enabled.
      requires_enroller_training = False, # enable if your enroller needs training

      multiple_model_scoring = 'average', # by default, compute the average between several models and the probe
      multiple_probe_scoring = 'average', # by default, compute the average between the model and several probes
      **kwargs                            # parameters from the derived class that should be reported in the __str__() function
  ):
    """Initializes the Tool.
    Call this constructor in derived class implementations.
    If your derived tool performs feature projection, please register this here.
    If it needs training for the projector or the enroller, please set this here, too.
    """

    self.performs_projection = performs_projection
    self.requires_projector_training = performs_projection and requires_projector_training
    self.split_training_features_by_client = split_training_features_by_client
    self.use_projected_features_for_enrollment = performs_projection and use_projected_features_for_enrollment
    self.requires_enroller_training = requires_enroller_training
    self.m_model_fusion_function = utils.score_fusion_strategy(multiple_model_scoring)
    self.m_probe_fusion_function = utils.score_fusion_strategy(multiple_probe_scoring)
    self._kwargs = kwargs
    self._kwargs.update({'multiple_model_scoring':multiple_model_scoring, 'multiple_probe_scoring':multiple_probe_scoring})


  def __str__(self):
    """This function returns all parameters of this class (and its derived class)."""
    return "%s(%s)" % (str(self.__class__), ", ".join(["%s=%s" % (key, value) for key,value in self._kwargs.items() if value is not None]))


  def enroll(self, enroll_features):
    """This function will enroll and return the model from the given list of features.
    It must be overwritten by derived classes.
    """
    raise NotImplementedError("Please overwrite this function in your derived class")


  def score(self, model, probe):
    """This function will compute the score between the given model and probe.
    It must be overwritten by derived classes.
    """
    raise NotImplementedError("Please overwrite this function in your derived class")


  def score_for_multiple_models(self, models, probe):
    """This function computes the score between the given model list and the given probe.
    In this base class implementation, it computes the scores for each model using the 'score' method,
    and fuses the scores using the fusion method specified in the constructor of this class.
    Usually this function is called from derived class 'score' functions."""
    if isinstance(models, list):
      return self.m_model_fusion_function([self.score(model, probe) for model in models])
    elif isinstance(models, numpy.ndarray):
      return self.m_model_fusion_function([self.score(models[i,:], probe) for i in range(models.shape[0])])
    else:
      raise ValueError("The model does not have the desired format (list, array, ...)")


  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and the given probe files.
    In this base class implementation, it computes the scores for each probe file using the 'score' method,
    and fuses the scores using the fusion method specified in the constructor of this class."""
    if isinstance(probes, list):
      return self.m_probe_fusion_function([self.score(model, probe) for probe in probes])
    else:
      # only one probe feature -> use the default scoring function
      return self.score(model, probes)


  ############################################################
  ### Special functions that might be overwritten on need
  ############################################################

  def save_feature(self, feature, feature_file):
    """Saves the given *projected* feature to a file with the given name.
    In this base class implementation:

    - If the given feature has a 'save' attribute, it calls feature.save(bob.io.base.HDF5File(feature_file), 'w').
      In this case, the given feature_file might be either a file name or a bob.io.base.HDF5File.
    - Otherwise, it uses bob.io.base.save to do that.

    If you have a different format, please overwrite this function.

    Please register 'performs_projection = True' in the constructor to enable this function.
    """
    utils.save(feature, feature_file)


  def read_feature(self, feature_file):
    """Reads the *projected* feature from file.
    In this base class implementation, it uses bob.io.base.load to do that.
    If you have different format, please overwrite this function.

    Please register 'performs_projection = True' in the constructor to enable this function.
    """
    return utils.load(feature_file)


  def save_model(self, model, model_file):
    """Saves the enrolled model to the given file.
    In this base class implementation:

    - If the given model has a 'save' attribute, it calls model.save(bob.io.base.HDF5File(model_file), 'w').
      In this case, the given model_file might be either a file name or a bob.io.base.HDF5File.
    - Otherwise, it uses bob.io.base.save to do that.

    If you have a different format, please overwrite this function.
    """
    utils.save(model, model_file)


  def read_model(self, model_file):
    """Loads the enrolled model from file.
    In this base class implementation, it uses bob.io.base.load to do that.

    If you have a different format, please overwrite this function.
    """
    return utils.load(model_file)


  def read_probe(self, probe_file):
    """Reads the probe feature from file.
    By default, the probe feature is identical to the projected feature.
    Hence, this base class implementation simply calls self.read_feature(...).

    If your tool requires different behavior, please overwrite this function.
    """
    return self.read_feature(probe_file)



  def train_projector(self, training_features, projector_file):
    """This function can be overwritten to train the feature projector.
    If you do this, please also register the function by calling this base class constructor
    and enabling the training by 'requires_projector_training = True'.

    The training function gets two parameters:

    - training_features: A list of *extracted* features that can be used for training the extractor.
    - projector_file: The file to write. This file should be readable with the 'load_projector' function (see above).
    """
    raise NotImplementedError("Please overwrite this function in your derived class, or unset the 'requires_projector_training' option in the constructor.")


  def load_projector(self, projector_file):
    """Loads the parameters required for feature projection from file.
    This function usually is only useful in combination with the 'train_projector' function (see above).
    In this base class implementation, it does nothing.

    Please register 'performs_projection = True' in the constructor to enable this function.
    """
    pass


  def train_enroller(self, training_features, enroller_file):
    """This function can be overwritten to train the model enroller.
    If you do this, please also register the function by calling this base class constructor
    and enabling the training by 'require_enroller_training = True'.

    The training function gets two parameters:

    - training_features: A dictionary of *extracted* or *projected* features, which are sorted by clients, that can be used for training the extractor.
    - enroller_file: The file to write. This file should be readable with the 'load_enroller' function (see above).
    """


  def load_enroller(self, enroller_file):
    """Loads the parameters required for model enrollment from file.
    This function usually is only useful in combination with the 'train_enroller' function (see above).
    This function is always called AFTER calling the 'load_projector'.
    In this base class implementation, it does nothing.
    """
    pass
