#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import os
import numpy
import bob
from .. import utils

class ToolChainZT:
  """This class includes functionalities for a default tool chain to produce verification scores"""

  def __init__(self, file_selector):
    """Initializes the tool chain object with the current file selector"""
    self.m_file_selector = file_selector

  def __read_original_image__(self, filename, preprocessor):
    """Loads the given image from file, using the function specified by the preprocessor, if available"""
    if hasattr(preprocessor, 'read_original_image'):
      return preprocessor.read_original_image(str(filename))
    else:
      return bob.io.load(str(filename))

  def __read_image__(self, filename, preprocessor):
    """Loads the given image from file, using the function specified by the preprocessor, if available"""
    if hasattr(preprocessor, 'read_image'):
      return preprocessor.read_image(str(filename))
    else:
      return bob.io.load(str(filename))

  def __save_image__(self, data, filename, preprocessor):
    """Saves the given image to the given filename, using the function specified by the preprocessor if available"""
    utils.ensure_dir(os.path.dirname(filename))
    if hasattr(preprocessor, 'save_image'):
      # Tool has a save_feature function, so use this one
      preprocessor.save_image(data, str(filename))
    elif hasattr(data, 'save'):
      # this is some class that supports saving itself
      data.save(bob.io.HDF5File(str(filename), "w"))
    else:
      # this is most probably a numpy.ndarray that can be saved by bob.io.save
      bob.io.save(data, str(filename))

  def __read_feature__(self, feature_file, tool):
    """This function reads the feature from file. It uses the tool.read_feature() function, if available, otherwise it uses bob.io.load()"""
    if hasattr(tool, 'read_feature'):
      return tool.read_feature(str(feature_file))
    else:
      return bob.io.load(str(feature_file))

  def __save_feature__(self, data, filename, tool):
    """Saves the feature to file using the function specified by the tool if available"""
    utils.ensure_dir(os.path.dirname(filename))
    if hasattr(tool, 'save_feature'):
      # Tool has a save_feature function, so use this one
      tool.save_feature(data, str(filename))
    elif hasattr(data, 'save'):
      # this is some class that supports saving itself
      data.save(bob.io.HDF5File(str(filename), "w"))
    else:
      # this is most probably a numpy.ndarray that can be saved by bob.io.save
      bob.io.save(data, str(filename))

  def __read_model__(self, model_file, tool):
    """This function reads the model from file. It uses the tool.read_model() function, if available, otherwise it uses bob.io.load()"""
    if hasattr(tool, 'read_model'):
      return tool.read_model(str(model_file))
    else:
      return bob.io.load(str(model_file))

  def __save_model__(self, data, filename, tool):
    """Saves the model using the function specified by the tool if available"""
    utils.ensure_dir(os.path.dirname(filename))
    # Tool has a save_model function, so use this one
    if hasattr(tool, 'save_model'):
      tool.save_model(data, str(filename))
    elif hasattr(data, 'save'):
      # this is some class that supports saving itself
      data.save(bob.io.HDF5File(str(filename), "w"))
    else:
      # this is most probably a numpy.ndarray that can be saved by bob.io.save
      bob.io.save(data, str(filename))

  def __read_probe__(self, probe_file, tool):
    """This function reads the probe from file. It uses the tool.read_probe() function, if available, otherwise it uses bob.io.load()"""
    if hasattr(tool, 'read_probe'):
      return tool.read_probe(str(probe_file))
    else:
      return bob.io.load(str(probe_file))


  def __check_file__(self, filename, force, expected_file_size = 1):
    """Checks if the file exists and has size greater or equal to expected_file_size.
    If the file is to small, or if the force option is set to true, the file is removed.
    This function returns true is the file is there, otherwise false"""
    if os.path.exists(filename):
      if force or os.path.getsize(filename) < expected_file_size:
        utils.debug("  .. Removing old file '%s'." % filename)
        os.remove(filename)
        return False
      else:
        return True
    return False



  def preprocess_images(self, preprocessor, indices=None, force=False):
    """Preprocesses the images with the given preprocessor"""
    # get the file lists
    image_files = self.m_file_selector.original_image_list()
    preprocessed_image_files = self.m_file_selector.preprocessed_image_list()

    # select a subset of keys to iterate
    keys = sorted(image_files.keys())
    if indices != None:
      keys = keys[indices[0]:indices[1]]
      utils.info("- Preprocessing: Splitting of index range %s" % indices)

    utils.info("- Preprocessing: processing %d images from directory '%s' to directory '%s'" % (len(keys), self.m_file_selector.m_config.image_directory, self.m_file_selector.m_config.preprocessed_dir))
    # iterate through the images and perform normalization

    # read eye files
    # - note: the resulting value of eye_files may be None
    annotation_list = self.m_file_selector.annotation_list()

    for k in keys:
      image = self.__read_original_image__(image_files[k], preprocessor)
      preprocessed_image_file = preprocessed_image_files[k]

      if not self.__check_file__(preprocessed_image_file, force):
        annotations = None
        if annotation_list != None:
          # read eyes position file
          annotations = utils.read_annotations(annotation_list[k], self.m_file_selector.m_db_options.annotation_type)

        # call the image preprocessor
        preprocessed_image = preprocessor(image, annotations)
        self.__save_image__(preprocessed_image, preprocessed_image_file, preprocessor)



  def __read_images__(self, files, preprocessor):
    retval = {}
    for k in files.keys():
      retval[k] = self.__read_image__(files[k], preprocessor)
    return retval

  def read_images_by_client__(self, files, preprocessor):
    retval = {}
    for client in files.keys():
      # images for the client
      data = {}
      client_files = files[client]
      for k in client_files.keys():
        data[k] = self.__read_image__(client_files[k], preprocessor)
      retval[client] = data
    return retval

  def train_extractor(self, extractor, preprocessor, force = False):
    """Trains the feature extractor, if it requires training"""
    if hasattr(extractor,'train'):
      extractor_file = self.m_file_selector.extractor_file()
      if self.__check_file__(extractor_file, force, 1000):
        utils.info("- Extraction: extractor '%s' already exists." % extractor_file)
      else:
        # read training files
        if hasattr(extractor, 'use_training_images_sorted_by_identity'):
          train_files = self.m_file_selector.training_feature_list_by_clients('preprocessed', 'train_extractor')
          train_images = self.__read_images_by_client__(train_files, preprocessor)
          utils.info("- Extraction: training extractor '%s' using %d identities: " %(extractor_file, len(train_files)))
        else:
          train_files = self.m_file_selector.training_image_list()
          train_images = self.__read_images__(train_files, preprocessor)
          utils.info("- Extraction: training extractor '%s' using %d training files: " %(extractor_file, len(train_files)))
        # train model
        extractor.train(train_images, extractor_file)



  def extract_features(self, extractor, preprocessor, indices = None, force=False):
    """Extracts the features using the given extractor"""
    if hasattr(extractor, 'load'):
      extractor.load(self.m_file_selector.extractor_file())
    image_files = self.m_file_selector.preprocessed_image_list()
    feature_files = self.m_file_selector.feature_list()

    # extract the features
    keys = sorted(image_files.keys())
    if indices != None:
      keys = keys[indices[0]:indices[1]]
      utils.info("- Extraction: splitting of index range %s" % indices)

    utils.info("- Extraction: extracting %d features from directory '%s' to directory '%s'" % (len(keys), self.m_file_selector.m_config.preprocessed_dir, self.m_file_selector.m_config.features_dir))
    for k in keys:
      image_file = image_files[k]
      feature_file = feature_files[k]

      if not self.__check_file__(feature_file, force):
        # load image
        image = self.__read_image__(image_file, preprocessor)
        # extract feature
        feature = extractor(image)
        # Save feature
        self.__save_feature__(feature, str(feature_file), extractor)



  def __read_features__(self, files, extractor):
    retval = {}
    for k in files.keys():
      retval[k] = self.__read_feature__(files[k], extractor)
    return retval

  def __read_features_by_client__(self, files, extractor):
    retval = {}
    for client in files.keys():
      # images for the client
      data = {}
      client_files = files[client]
      for k in client_files.keys():
        data[k] = self.__read_feature__(client_files[k], extractor)
      retval[client] = data
    return retval

  def train_projector(self, tool, extractor, force=False):
    """Train the feature projector with the extracted features of the world group"""
    if hasattr(tool,'train_projector'):
      projector_file = self.m_file_selector.projector_file()

      if self.__check_file__(projector_file, force, 1000):
        utils.info("- Projection: projector '%s' already exists." % projector_file)
      else:
        # train projector
        if hasattr(tool, 'use_training_features_sorted_by_identity'):
          train_files = self.m_file_selector.training_feature_list_by_clients('features', 'train_projector')
          train_features = self.__read_features_by_client__(train_files, extractor)
          utils.info("- Projection: training projector '%s' using %d identities: " %(projector_file, len(train_files)))
        else:
          train_files = self.m_file_selector.training_feature_list()
          train_features = self.__read_features__(train_files, extractor)
          utils.info("- Projection: training projector '%s' using %d training files: " %(projector_file, len(train_files)))

        # perform training
        tool.train_projector(train_features, str(projector_file))



  def project_features(self, tool, extractor, indices = None, force=False):
    """Extract the features for all files of the database"""
    # load the projector file
    if hasattr(tool, 'project'):
      if hasattr(tool, 'load_projector'):
        tool.load_projector(self.m_file_selector.projector_file())

      feature_files = self.m_file_selector.feature_list()
      projected_files = self.m_file_selector.projected_list()
      # extract the features
      keys = sorted(feature_files.keys())
      if indices != None:
        keys = keys[indices[0]:indices[1]]
        utils.info("- Projection: splitting of index range %s" % indices)

      utils.info("- Projection: projecting %d images from directory '%s' to directory '%s'" % (len(keys), self.m_file_selector.m_config.features_dir, self.m_file_selector.m_config.projected_dir))
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
          self.__save_feature__(projected, projected_file, tool)


  def train_enroller(self, tool, extractor, force=False):
    """Trains the model enroller using the extracted or projected features"""
    use_projected_features = hasattr(tool, 'project') and not hasattr(tool, 'use_unprojected_features_for_model_enroll')
    reader = tool if use_projected_features else extractor
    if hasattr(tool, 'train_enroller'):
      enroller_file = self.m_file_selector.enroller_file()

      if self.__check_file__(enroller_file, force, 1000):
        utils.info("- Enrollment: enroller '%s' already exists." % enroller_file)
      else:
        if hasattr(tool, 'load_projector'):
          tool.load_projector(self.m_file_selector.projector_file())
        # training models
        train_files = self.m_file_selector.training_feature_list_by_clients('projected' if use_projected_features else 'features', 'train_enroller')
        train_features = self.__read_features_by_client__(train_files, reader)

        # perform training
        utils.info("- Enrollment: training enroller '%s' using %d identities: " %(enroller_file, len(train_features)))
        tool.train_enroller(train_features, str(enroller_file))



  def enroll_models(self, tool, extractor, compute_zt_norm, indices = None, groups = ['dev', 'eval'], types = ['N','T'], force=False):
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
    reader = tool if use_projected_features else extractor

    # Create Models
    if 'N' in types:
      for group in groups:
        model_ids = self.m_file_selector.model_ids(group)

        if indices != None:
          model_ids = model_ids[indices[0]:indices[1]]
          utils.info("- Enrollment: splitting of index range %s" % indices)

        utils.info("- Enrollment: enrolling models of group '%s'" % group)
        for model_id in model_ids:
          # Path to the model
          model_file = self.m_file_selector.model_file(model_id, group)

          # Removes old file if required
          if not self.__check_file__(model_file, force):
            enroll_files = self.m_file_selector.enroll_files(model_id, group, use_projected_features)

            # load all files into memory
            enroll_features = []
            for k in sorted(enroll_files.keys()):
              # processes one file
              feature = self.__read_feature__(str(enroll_files[k]), reader)
              enroll_features.append(feature)

            model = tool.enroll(enroll_features)
            # save the model
            self.__save_model__(model, model_file, tool)

    # T-Norm-Models
    if 'T' in types and compute_zt_norm:
      for group in groups:
        model_ids = self.m_file_selector.tmodel_ids(group)

        if indices != None:
          model_ids = model_ids[indices[0]:indices[1]]
          utils.info("- Enrollment: splitting of index range %s" % indices)

        utils.info("- Enrollment: enrolling T-models of group '%s'" % group)
        for model_id in model_ids:
          # Path to the model
          model_file = self.m_file_selector.tmodel_file(model_id, group)

          # Removes old file if required
          if not self.__check_file__(model_file, force):
            enroll_files = self.m_file_selector.tenroll_files(model_id, group, use_projected_features)

            # load all files into memory
            enroll_features = []
            for k in sorted(enroll_files.keys()):
              # processes one file

              feature = self.__read_feature__(str(enroll_files[k]))
              enroll_features.append(feature)

            model = tool.enroll(enroll_features)
            # save model
            self.__save_model__(model, model_file, tool)



  def __scores__(self, model, probe_objects):
    """Compute simple scores for the given model"""
    scores = numpy.ndarray((1,len(probe_objects)), 'float64')

    # Loops over the probes
    i = 0
    for k in sorted(probe_objects.keys()):
      # read probe
      probe = self.__read_probe__(str(probe_objects[k][0]), self.m_tool)
      # compute score
      scores[0,i] = self.m_tool.score(model, probe)
      i += 1
    # Returns the scores
    return scores

  def __scores_preloaded__(self, model, probes):
    """Compute simple scores for the given model"""
    scores = numpy.ndarray((1,len(probes)), 'float64')

    # Loops over the probes
    i = 0
    for k in sorted(probes.keys()):
      # take pre-loaded probe
      probe = probes[k]
      # compute score
      scores[0,i] = self.m_tool.score(model, probe)
      i += 1
    # Returns the scores
    return scores


  def __probe_split__(self, selected_probe_objects, probes):
    """Helper function required when probe files are preloaded"""
    sel = 0
    res = {}
    # iterate over all probe files
    for k in (selected_probe_objects.keys()):
      # add probe
      res[k] = probes[k]

    # return the split database
    return res


  def __scores_a__(self, model_ids, group, compute_zt_norm, force, preload_probes):
    """Computes A scores. For non-ZT-norm, these are the only scores that are actually computed."""
    # preload the probe files for a faster access (and fewer network load)
    if preload_probes:
      utils.info("- Scoring: preloading probe files of group '%s'" % group)
      all_probe_objects = self.m_file_selector.probe_files(group, self.m_use_projected_dir)
      all_probes = {}
      # read all probe files into memory
      for k in sorted(all_probe_objects.keys()):
        all_probes[k] = self.__read_probe__(str(all_probe_objects[k][0]), self.m_tool)

    if compute_zt_norm:
      utils.info("- Scoring: computing score matrix A for group '%s'" % group)
    else:
      utils.info("- Scoring: computing scores for group '%s'" % group)

    # Computes the raw scores for each model
    for model_id in model_ids:
      # test if the file is already there
      score_file = self.m_file_selector.a_file(model_id, group) if compute_zt_norm else self.m_file_selector.no_norm_file(model_id, group)
      if self.__check_file__(score_file, force):
        utils.warn("score file '%s' already exists." % (score_file))
      else:
        # get the probe split
        probe_objects = self.m_file_selector.probe_files_for_model(model_id, group, self.m_use_projected_dir)
        model = self.__read_model__(self.m_file_selector.model_file(model_id, group), self.m_tool)
        if preload_probes:
          # select the probe files for this model from all probes
          current_probes = self.__probe_split__(probe_objects, all_probes)
          # compute A matrix
          a = self.__scores_preloaded__(model, current_probes)
        else:
          a = self.__scores__(model, probe_objects)

        if compute_zt_norm:
          # write A matrix only when you want to compute zt norm afterwards
          bob.io.save(a, self.m_file_selector.a_file(model_id, group))

        # Saves to text file
        scores_list = utils.convertScoreToList(numpy.reshape(a,a.size), probe_objects)
        f_nonorm = open(self.m_file_selector.no_norm_file(model_id, group), 'w')
        for x in scores_list:
          f_nonorm.write(str(x[2]) + " " + str(x[1]) + " " + str(x[3]) + " " + str(x[4]) + "\n")
        f_nonorm.close()

  def __scores_b__(self, model_ids, group, force, preload_probes):
    """Computes B scores"""
    # probe files:
    zprobe_objects = self.m_file_selector.zprobe_files(group, self.m_use_projected_dir)
    # preload the probe files for a faster access (and fewer network load)
    if preload_probes:
      utils.info("- Scoring: preloading Z-probe files of group '%s'" % group)
      zprobes = {}
      # read all probe files into memory
      for k in sorted(zprobe_objects.keys()):
        zprobes[k] = self.__read_probe__(str(zprobe_objects[k][0]), self.m_tool)

    utils.info("- Scoring: computing score matrix B for group '%s'" % group)

    # Loads the models
    for model_id in model_ids:
      # test if the file is already there
      score_file = self.m_file_selector.b_file(model_id, group)
      if self.__check_file__(score_file, force):
        utils.warn("score file '%s' already exists." % (score_file))
      else:
        model = self.__read_model__(self.m_file_selector.model_file(model_id, group), self.m_tool)
        if preload_probes:
          b = self.__scores_preloaded__(model, zprobes)
        else:
          b = self.__scores__(model, zprobe_objects)
        bob.io.save(b, score_file)

  def __scores_c__(self, tmodel_ids, group, force, preload_probes):
    """Computes C scores"""
    # probe files:
    probe_objects = self.m_file_selector.probe_files(group, self.m_use_projected_dir)

    # preload the probe files for a faster access (and fewer network load)
    if preload_probes:
      utils.info("- Scoring: preloading probe files of group '%s'" % group)
      probes = {}
      # read all probe files into memory
      for k in sorted(probe_objects.keys()):
        probes[k] = self.__read_probe__(str(probe_objects[k][0]), self.m_tool)

    utils.info("- Scoring: computing score matrix C for group '%s'" % group)

    # Computes the raw scores for the T-Norm model
    for tmodel_id in tmodel_ids:
      # test if the file is already there
      score_file = self.m_file_selector.c_file(tmodel_id, group)
      if self.__check_file__(score_file, force):
        utils.warn("score file '%s' already exists." % (score_file))
      else:
        tmodel = self.__read_model__(self.m_file_selector.tmodel_file(tmodel_id, group), self.m_tool)
        if preload_probes:
          c = self.__scores_preloaded__(tmodel, probes)
        else:
          c = self.__scores__(tmodel, probe_objects)
        bob.io.save(c, score_file)

  def __scores_d__(self, tmodel_ids, group, force, preload_probes):
    """Computes D scores"""
    # probe files:
    zprobe_objects = self.m_file_selector.zprobe_files(group, self.m_use_projected_dir)
    # preload the probe files for a faster access (and fewer network load)
    if preload_probes:
      utils.info("- Scoring: preloading Z-probe files of group '%s'" % group)
      zprobes = {}
      # read all probe files into memory
      for k in sorted(zprobe_objects.keys()):
        zprobes[k] = self.__read_probe__(str(zprobe_objects[k][0]), self.m_tool)

    utils.info("- Scoring: computing score matrix D for group '%s'" % group)

    # Gets the Z-Norm impostor samples
    zprobe_ids = []
    for k in sorted(zprobe_objects.keys()):
      zprobe_ids.append(zprobe_objects[k][3])

    # Loads the T-Norm models
    for tmodel_id in tmodel_ids:
      # test if the file is already there
      score_file = self.m_file_selector.d_same_value_file(tmodel_id, group)
      if self.__check_file__(score_file, force):
        utils.warn("score file '%s' already exists." % (score_file))
      else:
        tmodel = self.__read_model__(self.m_file_selector.tmodel_file(tmodel_id, group), self.m_tool)
        if preload_probes:
          d = self.__scores_preloaded__(tmodel, zprobes)
        else:
          d = self.__scores__(tmodel, zprobe_objects)
        bob.io.save(d, self.m_file_selector.d_file(tmodel_id, group))

        tclient_id = [self.m_file_selector.m_config.db.get_client_id_from_model_id(tmodel_id)]
        d_same_value_tm = bob.machine.ztnorm_same_value(tclient_id, zprobe_ids)
        bob.io.save(d_same_value_tm, score_file)


  def compute_scores(self, tool, compute_zt_norm, force = False, indices = None, groups = ['dev', 'eval'], types = ['A', 'B', 'C', 'D'], preload_probes = False):
    """Computes the scores for the given groups (by default 'dev' and 'eval')"""
    # save tool for internal use
    self.m_tool = tool
    self.m_use_projected_dir = hasattr(tool, 'project')

    # load the projector, if needed
    if hasattr(tool,'load_projector'):
      tool.load_projector(self.m_file_selector.projector_file())
    if hasattr(tool,'load_enroller'):
      tool.load_enroller(self.m_file_selector.enroller_file())

    for group in groups:
      # get model ids
      model_ids = self.m_file_selector.model_ids(group)
      if compute_zt_norm:
        tmodel_ids = self.m_file_selector.tmodel_ids(group)

      # compute A scores
      if 'A' in types:
        if indices != None:
          model_ids_short = model_ids[indices[0]:indices[1]]
        else:
          model_ids_short = model_ids
        self.__scores_a__(model_ids_short, group, compute_zt_norm, force, preload_probes)

      if compute_zt_norm:
        # compute B scores
        if 'B' in types:
          if indices != None:
            model_ids_short = model_ids[indices[0]:indices[1]]
          else:
            model_ids_short = model_ids
          self.__scores_b__(model_ids_short, group, force, preload_probes)

        # compute C scores
        if 'C' in types:
          if indices != None:
            tmodel_ids_short = tmodel_ids[indices[0]:indices[1]]
          else:
            tmodel_ids_short = tmodel_ids
          self.__scores_c__(tmodel_ids_short, group, force, preload_probes)

        # compute D scores
        if 'D' in types:
          if indices != None:
            tmodel_ids_short = tmodel_ids[indices[0]:indices[1]]
          else:
            tmodel_ids_short = tmodel_ids
          self.__scores_d__(tmodel_ids_short, group, force, preload_probes)



  def __scores_c_normalize__(self, model_ids, tmodel_ids, group):
    """Compute normalized probe scores using T-model scores"""
    # read all tmodel scores
    c_for_all = None
    for tmodel_id in tmodel_ids:
      tmp = bob.io.load(self.m_file_selector.c_file(tmodel_id, group))
      if c_for_all == None:
        c_for_all = tmp
      else:
        c_for_all = numpy.vstack((c_for_all, tmp))
    # iterate over all models and generate C matrices for that specific model
    probe_objects = self.m_file_selector.probe_files(group, False)
    for model_id in model_ids:
      # select the correct probe files for the current model
      model_probes = self.m_file_selector.probe_files_for_model(model_id, group, False)
      probes_used = utils.probes_used_generate_vector(probe_objects, model_probes)
      c_for_model = utils.probes_used_extract_scores(c_for_all, probes_used)
      # Save C matrix to file
      bob.io.save(c_for_model, self.m_file_selector.c_file_for_model(model_id, group))

  def __scores_d_normalize__(self, tmodel_ids, group):
    """Compute normalized D scores for the given T-model ids"""
    # initialize D and D_same_value matrices
    d_for_all = None
    d_same_value = None
    for tmodel_id in tmodel_ids:
      tmp = bob.io.load(self.m_file_selector.d_file(tmodel_id, group))
      tmp2 = bob.io.load(self.m_file_selector.d_same_value_file(tmodel_id, group))
      if d_for_all == None and d_same_value == None:
        d_for_all = tmp
        d_same_value = tmp2
      else:
        d_for_all = numpy.vstack((d_for_all, tmp))
        d_same_value = numpy.vstack((d_same_value, tmp2))

    # Saves to files
    bob.io.save(d_for_all, self.m_file_selector.d_matrix_file(group))
    bob.io.save(d_same_value, self.m_file_selector.d_same_value_matrix_file(group))



  def zt_norm(self, groups = ['dev', 'eval']):
    """Computes ZT-Norm using the previously generated A, B, C, and D files"""
    for group in groups:
      utils.info("- Scoring: computing ZT-norm for group '%s'" % group)
      # list of models
      model_ids = self.m_file_selector.model_ids(group)
      tmodel_ids = self.m_file_selector.tmodel_ids(group)

      # first, normalize C and D scores
      self.__scores_c_normalize__(model_ids, tmodel_ids, group)
      # and normalize it
      self.__scores_d_normalize__(tmodel_ids, group)


      # load D matrices only once
      d = bob.io.load(self.m_file_selector.d_matrix_file(group))
      d_same_value = bob.io.load(self.m_file_selector.d_same_value_matrix_file(group)).astype(bool)
      # Loops over the model ids
      for model_id in model_ids:
        # Loads probe files to get information about the type of access
        probe_objects = self.m_file_selector.probe_files_for_model(model_id, group, False)

        # Loads A, B, C, D and D_same_value matrices
        a = bob.io.load(self.m_file_selector.a_file(model_id, group))
        b = bob.io.load(self.m_file_selector.b_file(model_id, group))
        c = bob.io.load(self.m_file_selector.c_file_for_model(model_id, group))

        # compute zt scores
        ztscores_m = bob.machine.ztnorm(a, b, c, d, d_same_value)

        # Saves to text file
        ztscores_list = utils.convertScoreToList(numpy.reshape(ztscores_m, ztscores_m.size), probe_objects)
        sc_ztnorm_filename = self.m_file_selector.zt_norm_file(model_id, group)
        f_ztnorm = open(sc_ztnorm_filename, 'w')
        for x in ztscores_list:
          f_ztnorm.write(str(x[2]) + " " + str(x[1]) + " " + str(x[3]) + " " + str(x[4]) + "\n")
        f_ztnorm.close()



  def concatenate(self, compute_zt_norm, groups = ['dev', 'eval']):
    """Concatenates all results into one (or two) score files per group."""
    for group in groups:
      utils.info("- Scoring: concatenating score files for group '%s'" % group)
      # (sorted) list of models
      model_ids = self.m_file_selector.model_ids(group)

      f = open(self.m_file_selector.no_norm_result_file(group), 'w')
      # Concatenates the scores
      for model_id in model_ids:
        model_file = self.m_file_selector.no_norm_file(model_id, group)
        if not os.path.exists(model_file):
          raise IOError("The score file '%s' cannot be found. Aborting!" % model_file)

        res_file = open(model_file, 'r')
        f.write(res_file.read())
      f.close()

      if compute_zt_norm:
        f = open(self.m_file_selector.zt_norm_result_file(group), 'w')
        # Concatenates the scores
        for model_id in model_ids:
          res_file = open(self.m_file_selector.zt_norm_file(model_id, group), 'r')
          f.write(res_file.read())
        f.close()
