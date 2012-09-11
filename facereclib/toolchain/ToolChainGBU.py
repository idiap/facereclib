#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import os
import numpy
import bob
from .. import utils

class ToolChainGBU:
  """This class includes functionalities for a default tool chain to produce verification scores"""

  def __init__(self, file_selector):
    """Initializes the tool chain object with the current file selector"""
    self.m_file_selector = file_selector

  def __save_feature__(self, data, filename):
    """Saves the given feature to the given file"""
    utils.ensure_dir(os.path.dirname(filename))
    if hasattr(self.m_tool, 'save_feature'):
      # Tool has a save_feature function, so use this one
      self.m_tool.save_feature(data, str(filename))
    elif hasattr(data, 'save'):
      # this is some class that supports saving itself
      data.save(bob.io.HDF5File(str(filename), "w"))
    else:
      # this is most probably a numpy.ndarray that can be saved by bob.io.save
      bob.io.save(data, str(filename))

  def __save_model__(self, data, filename, tool=None):
    utils.ensure_dir(os.path.dirname(filename))
    if tool == None:
      tool = self.m_tool
    # Tool has a save_model function, so use this one
    if hasattr(tool, 'save_model'):
      tool.save_model(data, str(filename))
    elif hasattr(data, 'save'):
      # this is some class that supports saving itself
      data.save(bob.io.HDF5File(str(filename), "w"))
    else:
      # this is most probably a numpy.ndarray that can be saved by bob.io.save
      bob.io.save(data, str(filename))


  def __check_file__(self, filename, force, expected_file_size = 1):
    """Checks if the file exists and has size greater or equal to expected_file_size.
    If the file is to small, or if the force option is set to true, the file is removed.
    This function returns true is the file is there, otherwise false"""
    if os.path.exists(filename):
      if force or os.path.getsize(filename) < expected_file_size:
        print "Removing old file '%s'." % filename
        os.remove(filename)
        return False
      else:
        return True
    return False



  def preprocess_images(self, preprocessor, sets=['training','target','query'], indices=None, force=False):
    """Preprocesses the images with the given preprocessing tool"""
    for set in sets:
      # get the file lists
      image_files = self.m_file_selector.original_image_list(set)
      # read eye files
      annotation_list = self.m_file_selector.annotation_list(set)
      preprocessed_image_files = self.m_file_selector.preprocessed_image_list(set)

      # select a subset of keys to iterate
      keys = sorted(image_files.keys())
      if indices != None:
        keys = keys[indices[0]:indices[1]]
        print "Splitting of index range", indices, "to",

      print "preprocess", len(keys), "images from directory", os.path.dirname(image_files.values()[0]), "to directory", os.path.dirname(preprocessed_image_files.values()[0])
      # iterate through the images and perform normalization
      for k in keys:
        image_file = image_files[k]
        preprocessed_image_file = preprocessed_image_files[k]
        annotations = annotation_list[k]

        if not self.__check_file__(preprocessed_image_file, force):
          # read eyes position file
          utils.ensure_dir(os.path.dirname(preprocessed_image_file))
          preprocessed_image = preprocessor(str(image_file), str(preprocessed_image_file), annotations)



  def train_extractor(self, extractor, force = False):
    """Trains the feature extractor, if it requires training"""
    if hasattr(extractor,'train'):
      extractor_file = self.m_file_selector.extractor_file()
      if self.__check_file__(extractor_file, force, 1000):
        print "Extractor '%s' already exists." % extractor_file
      else:
        # train model
        if hasattr(extractor, 'use_training_images_sorted_by_identity'):
          train_files = self.m_file_selector.training_feature_list_by_clients('preprocessed')
          print "Training Extractor '%s' using %d identities: " %(extractor_file, len(train_files))
        else:
          train_files = self.m_file_selector.training_feature_list('preprocessed')
          print "Training Extractor '%s' using %d training files: " %(extractor_file, len(train_files))
        extractor.train(train_files, extractor_file)



  def extract_features(self, extractor, sets=['training','target','query'], indices = None, force=False):
    """Extracts the features using the given extractor"""
    self.m_tool = extractor
    if hasattr(extractor, 'load'):
      extractor.load(self.m_file_selector.extractor_file())

    for set in sets:
      image_files = self.m_file_selector.preprocessed_image_list(set)
      feature_files = self.m_file_selector.feature_list(set)

      # extract the features
      keys = sorted(image_files.keys())
      if indices != None:
        keys = keys[indices[0]:indices[1]]
        print "Splitting of index range", indices, "to",

      print "extract", len(keys), "features from image directory", os.path.dirname(image_files.values()[0]), "to directory", os.path.dirname(feature_files.values()[0])
      for k in keys:
        image_file = image_files[k]
        feature_file = feature_files[k]

        if not self.__check_file__(feature_file, force):
          # load image
          image = bob.io.load(str(image_file))
          # extract feature
          feature = extractor(image)
           # Save feature
          self.__save_feature__(feature, str(feature_file))



  def train_projector(self, tool, force=False):
    """Train the feature extraction process with the preprocessed images of the world group"""
    if hasattr(tool,'train_projector'):
      projector_file = self.m_file_selector.projector_file()

      if self.__check_file__(projector_file, force, 1000):
        print "Projector '%s' already exists." % projector_file
      else:
        # train projector
        if hasattr(tool, 'use_training_features_sorted_by_identity'):
          train_files = self.m_file_selector.training_feature_list_by_clients('features')
          print "Training Projector '%s' using %d identities: " %(projector_file, len(train_files))
        else:
          train_files = self.m_file_selector.training_feature_list('features')
          print "Training Projector '%s' using %d training files: " %(projector_file, len(train_files))

        # perform training
        tool.train_projector(train_files, str(projector_file))



  def project_features(self, tool, extractor, sets=['training','target','query'], indices = None, force=False):
    """Extract the features for all files of the database"""
    self.m_tool = tool
    # load the projector file
    if hasattr(tool, 'project'):
      if hasattr(tool, 'load_projector'):
        tool.load_projector(self.m_file_selector.projector_file())

      for set in sets:
        feature_files = self.m_file_selector.feature_list(set)
        projected_files = self.m_file_selector.projected_list(set)

        # extract the features
        keys = sorted(feature_files.keys())
        if indices != None:
          keys = keys[indices[0]:indices[1]]
          print "Splitting of index range", indices, "to",

        print "project", len(keys), "features from directory", os.path.dirname(feature_files.values()[0]), "to directory", os.path.dirname(projected_files.values()[0])
        for k in keys:
          feature_file = feature_files[k]
          projected_file = projected_files[k]

          if not self.__check_file__(projected_file, force):
            # load feature
            feature = self.__read_feature__(feature_file, extractor)
            # project feature
            projected = tool.project(feature)
            # write it
            utils.ensure_dir(os.path.dirname(projected_file))
            self.__save_feature__(projected, projected_file)



  def __read_feature__(self, feature_file, tool = None):
    """This function reads the model from file. Overload this function if your model is no numpy.ndarray."""
    if not tool:
      tool = self.m_tool
    if hasattr(tool, 'read_feature'):
      return tool.read_feature(str(feature_file))
    else:
      return bob.io.load(str(feature_file))

  def train_enroller(self, tool, force=False):
    """Trains the model enrollment stage using the projected features"""
    self.m_tool = tool
    use_projected_features = hasattr(tool, 'project') and not hasattr(tool, 'use_unprojected_features_for_model_enroll')
    if hasattr(tool, 'train_enroller'):
      enroller_file = self.m_file_selector.enroller_file()

      if self.__check_file__(enroller_file, force, 1000):
        print "Enroller '%s' already exists." % enroller_file
      else:
        if hasattr(tool, 'load_projector'):
          tool.load_projector(self.m_file_selector.projector_file())
        # training models
        train_files = self.m_file_selector.training_feature_list_by_clients('projected' if use_projected_features else 'features')

        # perform training
        print "Training Enroller '%s' using %d identities: " %(enroller_file, len(train_files))
        tool.train_enroller(train_files, str(enroller_file))



  def enroll_models(self, tool, extractor, indices = None, force=False):
    """Enroll the models for 'dev' and 'eval' groups, for both models and T-Norm-models.
       This function by default used the projected features to compute the models.
       If you need unprojected features for the model, please define a variable with the name
       use_unprojected_features_for_model_enroll"""

    # read the projector file, if needed
    if hasattr(tool,'load_projector'):
      # read the feature extraction model
      tool.load_projector(self.m_file_selector.projector_file())
    if hasattr(tool, 'load_enroller'):
      # read the model enrollment file
      tool.load_enroller(self.m_file_selector.enroller_file())

    # use projected or unprojected features for model enrollment?
    use_projected_features = hasattr(tool, 'project') and not hasattr(tool, 'use_unprojected_features_for_model_enroll')
    # which tool to use to read the features...
    self.m_tool = tool if use_projected_features else extractor

    # enroll models
    model_ids = self.m_file_selector.model_ids()

    if indices != None:
      model_ids = model_ids[indices[0]:indices[1]]
      print "Splitting of index range", indices, "for",

    print "enrolling models"
    for model_id in model_ids:
      model_file = self.m_file_selector.model_file(model_id)

      # Removes old file if required
      if not self.__check_file__(model_file, force):
        enroll_files = self.m_file_selector.enroll_files(model_id, use_projected_features)

        # load all files into memory
        enroll_features = []
        for enroll_file in enroll_files.itervalues():
          # processes one file
          feature = self.__read_feature__(str(enroll_file))
          enroll_features.append(feature)

        model = tool.enroll(enroll_features)
        # save the model
        self.__save_model__(model, model_file, tool)



  def __read_model__(self, model_file):
    """This function reads the model from file. Overload this function if your model is no numpy.ndarray."""
    if hasattr(self.m_tool, 'read_model'):
      return self.m_tool.read_model(str(model_file))
    else:
      return bob.io.load(str(model_file))

  def __read_probe__(self, probe_file):
    """This function reads the probe from file. Overload this function if your probe is no numpy.ndarray."""
    if hasattr(self.m_tool, 'read_probe'):
      return self.m_tool.read_probe(str(probe_file))
    else:
      return bob.io.load(str(probe_file))

  def compute_scores(self, tool, force = False, indices = None, preload_probes = False):
    """Computes the scores between target and query"""
    # save tool for internal use
    self.m_tool = tool
    use_projected_dir = hasattr(tool, 'project')

    # load the projector, if needed
    if hasattr(tool,'load_projector'):
      tool.load_projector(self.m_file_selector.projector_file())
    if hasattr(tool,'load_enroller'):
      tool.load_enroller(self.m_file_selector.enroller_file())

    # get model and probe files
    model_ids = self.m_file_selector.model_ids()
    probe_files = self.m_file_selector.probe_files(use_projected_dir)

    # get the claimed probe ids
    claimed_probe_ids = {}
    for probe_id in probe_files:
      claimed_probe_ids[probe_id] = self.m_file_selector.client_id_for_id(probe_id)

    # preload probe files?
    if preload_probes:
      print "Preloading query files"
      preloaded_probes = {}
      for k, probe_file in probe_files.iteritems():
        preloaded_probes[k] = self.__read_probe__(str(probe_file))

    # use only some models?
    if indices != None:
      model_ids = model_ids[indices[0]:indices[1]]

    # compute scores.
    print "computing scores for", len(model_ids), "models"
    for model_id in model_ids:
      real_client_id = self.m_file_selector.client_id_for_id(model_id)

      score_file = str(self.m_file_selector.model_score_file(model_id))
      if self.__check_file__(score_file, force):
        print "Score file '%s' already exists." % (score_file)
      else:
        # load target (which is the model in our case)
        model = self.__read_model__(str(self.m_file_selector.model_file(model_id)))

        # compute scores for the current target
        score_list = []
        for probe_id in sorted(probe_files.keys()):
          # get query information
          if preload_probes:
            probe = preloaded_probes[probe_id]
          else:
            probe = self.__read_probe__(probe_files[probe_id])

          score = tool.score(model, probe)

          score_list.append((real_client_id, claimed_probe_ids[probe_id], score))

        # write score file
        f = open(score_file, 'w')
        for x in score_list:
          f.write(str(x[0]) + " " + str(x[1]) + " " + "None" + " " + str(x[2]) + "\n")
        f.close()



  def concatenate(self):
    """Concatenates all results into one score file"""
    # list of model indices
    model_ids = self.m_file_selector.model_ids()

    f = open(self.m_file_selector.result_file(), 'w')
    # Concatenates the scores
    for model_id in model_ids:
      model_file = self.m_file_selector.model_score_file(model_id)
      assert os.path.exists(model_file)
      res_file = open(model_file, 'r')
      f.write(res_file.read())
      res_file.close()

    f.close()
