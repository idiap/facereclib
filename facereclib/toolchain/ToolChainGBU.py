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
    
  def __save__(self, data, filename):
    utils.ensure_dir(os.path.dirname(filename))
    if hasattr(data, 'save'):
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
      image_files = self.m_file_selector.image_list(set)
      # read eye files
      eye_positions = self.m_file_selector.eye_position_list(set)
      preprocessed_image_files = self.m_file_selector.preprocessed_image_list(set)
  
      # select a subset of keys to iterate    
      partition = range(indices[0],indices[1]) if indices else range(len(image_files))
  
      print "preprocessing", len(partition), "images from directory", os.path.dirname(image_files[0]), "to directory", os.path.dirname(preprocessed_image_files[0])
      # iterate through the images and perform normalization
      for i in partition:
        image_file = image_files[i]
        preprocessed_image_file = preprocessed_image_files[i]
        eye_position = eye_positions[i]
  
        if not self.__check_file__(preprocessed_image_file, force):
          # read eyes position file
          utils.ensure_dir(os.path.dirname(preprocessed_image_file))
          preprocessed_image = preprocessor(str(image_file), eye_position, str(preprocessed_image_file))
              
  
  def train_extractor(self, extractor, force = False):
    """Trains the feature extractor, if it requires training"""
    if hasattr(extractor,'train'):
      extractor_file = self.m_file_selector.extractor_file()
      if self.__check_file__(extractor_file, force, 1000):
        print "Extractor '%s' already exists." % extractor_file
      else:
        # train model
        if hasattr(extractor, 'use_training_images_sorted_by_identity'):
          train_files = self.m_file_selector.training_feature_list_by_models('preprocessed')
          print "Training Extractor '%s' using %d identities: " %(extractor_file, len(train_files))
        else:
          train_files = self.m_file_selector.training_feature_list('preprocessed')
          print "Training Extractor '%s' using %d training files: " %(extractor_file, len(train_files))
        extractor.train(train_files, extractor_file)

  
  def extract_features(self, extractor, sets=['training','target','query'], indices = None, force=False):
    """Extracts the features using the given extractor"""
    if hasattr(extractor, 'load'):
      extractor.load(self.m_file_selector.extractor_file())
      
    for set in sets:
      image_files = self.m_file_selector.preprocessed_image_list(set)
      feature_files = self.m_file_selector.feature_list(set)
  
      # extract the features
      partition = range(indices[0],indices[1]) if indices else range(len(image_files))
  
      print "extracting", len(partition), "features from image directory", os.path.dirname(image_files[0]), "to directory", os.path.dirname(feature_files[0])
      for i in partition:
        image_file = image_files[i]
        feature_file = feature_files[i]
        
        if not self.__check_file__(feature_file, force):
          # load image
          image = bob.io.load(str(image_file))
          # extract feature
          feature = extractor(image)
          # Save feature
          self.__save__(feature, str(feature_file))
          
#### write feature to text file
          if False:
            f = open("/scratch/mguenther/GBU/Good/features_mine/"+str(os.path.splitext(os.path.basename(feature_file))[0])+'.txt', "w")
            for x in feature:
              f.write(str(x) + '\n')
            f.close()
####
      

  def train_projector(self, tool, force=False):
    """Train the feature extraction process with the preprocessed images of the world group"""
    if hasattr(tool,'train_projector'):
      projector_file = self.m_file_selector.projector_file()
      
      if self.__check_file__(projector_file, force, 1000):
        print "Projector '%s' already exists." % projector_file
      else:
        # train projector
        if hasattr(tool, 'use_training_features_sorted_by_identity'):
          train_files = self.m_file_selector.training_feature_list_by_models('features')
          print "Training Projector '%s' using %d identities: " %(projector_file, len(train_files))
        else:
          train_files = self.m_file_selector.training_feature_list('features')
          print "Training Projector '%s' using %d training files: " %(projector_file, len(train_files))

        # perform training
        tool.train_projector(train_files, str(projector_file))
 
    
  def project_features(self, tool, sets=['training','target','query'], indices = None, force=False):
    """Extract the features for all files of the database"""
    # load the projector file
    if hasattr(tool, 'project'):
      if hasattr(tool, 'load_projector'):
        tool.load_projector(self.m_file_selector.projector_file())
      
      for set in sets:
        feature_files = self.m_file_selector.feature_list(set)
        projected_files = self.m_file_selector.projected_list(set)

        # project the features
        partition = range(indices[0],indices[1]) if indices else range(len(feature_files))
  
        print "project", len(partition), "features from directory", os.path.dirname(feature_files[0]), "to directory", os.path.dirname(projected_files[0])
        for i in partition:
          feature_file = feature_files[i]
          projected_file = projected_files[i]
          
          if not self.__check_file__(projected_file, force):
            # load feature
            feature = bob.io.load(str(feature_file))
            # project feature
            projected = tool.project(feature)
            # write it
            utils.ensure_dir(os.path.dirname(projected_file))
            self.__save__(projected, projected_file)
  
    
  def __read_feature__(self, feature_file):
    """This function reads the model from file. Overload this function if your model is no numpy.ndarray."""
    if hasattr(self.m_tool, 'read_feature'):
      return self.m_tool.read_feature(feature_file)
    else:
      return bob.io.load(feature_file)
  
  def train_enroler(self, tool, force=False):
    """Traines the model enrolment stage using the projected features"""
    self.m_tool = tool
    use_projected_features = hasattr(tool, 'project') and not hasattr(tool, 'use_unprojected_features_for_model_enrol')
    if hasattr(tool, 'train_enroler'):
      enroler_file = self.m_file_selector.enroler_file()
      
      if self.__check_file__(enroler_file, force, 1000):
        print "Enroler '%s' already exists." % enroler_file
      else:
        if hasattr(tool, 'load_projector'):
          tool.load_projector(self.m_file_selector.projector_file())
        # training models
        train_files = self.m_file_selector.training_feature_list_by_models('projected' if use_projected_features else 'features')
  
        # perform training
        print "Training Enroler '%s' using %d identities: " %(enroler_file, len(train_files))
        tool.train_enroler(train_files, str(enroler_file))

    
  def enrol_models(self, tool, indices = None, force=False):
    """Enrol the models for 'dev' and 'eval' groups, for both models and T-Norm-models.
       This function by default used the projected features to compute the models.
       If you need unprojected features for the model, please define a variable with the name 
       use_unprojected_features_for_model_enrol"""
    
    self.m_tool = tool
    # read the projector file, if needed
    if hasattr(tool,'load_projector'):
      # read the feature extraction model
      tool.load_projector(self.m_file_selector.projector_file())
    if hasattr(tool, 'load_enroler'):
      # read the model enrolment file
      tool.load_enroler(self.m_file_selector.enroler_file())
      

    use_projected_features = hasattr(tool, 'project') and not hasattr(tool, 'use_unprojected_features_for_model_enrol')

    # enrol models
    model_indices = self.m_file_selector.model_indices()

    partition = range(indices[0],indices[1]) if indices else range(len(model_indices))

    print "enroling", len(partition), "models"
    for model_index in partition:
      model_file = self.m_file_selector.model_file(model_index)
      
      # Removes old file if required
      if not self.__check_file__(model_file, force):
        enrol_files = self.m_file_selector.enrol_files(model_index, use_projected_features)
            
        # load all files into memory
        enrol_features = []
        for enrol_index in range(len(enrol_files)):
          # processes one file
          feature = self.__read_feature__(str(enrol_files[enrol_index]))
          enrol_features.append(feature)
        
        model = tool.enrol(enrol_features)
        # save the model
        self.__save__(model, model_file)


  def __read_model__(self, model_file):
    """This function reads the model from file. Overload this function if your model is no numpy.ndarray."""
    if hasattr(self.m_tool, 'read_model'):
      return self.m_tool.read_model(model_file)
    else:
      return bob.io.load(model_file)
    
  def __read_probe__(self, probe_file):
    """This function reads the probe from file. Overload this function if your probe is no numpy.ndarray."""
    if hasattr(self.m_tool, 'read_probe'):
      return self.m_tool.read_probe(probe_file)
    else:
      return bob.io.load(probe_file)


  def compute_scores(self, tool, force = False, indices = None, preload_probes = False):
    """Computes the scores between target and query"""
    # save tool for internal use
    self.m_tool = tool
    use_projected_dir = hasattr(tool, 'project')
    
    # load the projector, if needed
    if hasattr(tool,'load_projector'):
      tool.load_projector(self.m_file_selector.projector_file())
    if hasattr(tool,'load_enroler'):
      tool.load_enroler(self.m_file_selector.enroler_file())

    model_indices = self.m_file_selector.model_indices()
    query_list = self.m_file_selector.probe_files(use_projected_dir)
    
    if preload_probes:
      print "Preloading query files"
      preloaded_probes = []
      for probe_file in query_list:
        preloaded_probes.append(self.__read_probe__(str(probe_file))) 
    
    # get the subset of target files to be used
    partition = range(indices[0],indices[1]) if indices else range(len(model_indices))
    
    print "computing scores for", len(partition), "models"
    for model_index in partition:

      score_file = str(self.m_file_selector.model_score_file(model_index))
      if self.__check_file__(score_file, force):
        print "Score file '%s' already exists." % (score_file)
      else:
        # load target (which is the model in our case)
        model = self.__read_model__(str(self.m_file_selector.model_file(model_index)))
        
        # get query files for the current target
        query_informations = self.m_file_selector.queries_for_target(model_index)
        
        # compute scores for the current target
        score_list = []
        for query_index in range(len(query_informations)):
          # get query information
          query_info = query_informations[query_index]
          if preload_probes:
            query = preloaded_probes[query_info[0]]
          else:
            query = self.__read_probe__(str(query_list[query_info[0]]))
        
          score = tool.score(model, query)
          
          score_list.append((query_info[2], query_info[1], score))
          
        # write score file
        f = open(score_file, 'w')
        for x in score_list:
          f.write(str(x[0]) + " " + str(x[1]) + " " + "None" + " " + str(x[2]) + "\n")
        f.close()
        

  def concatenate(self):
    """Concatenates all results into one score file"""
    # list of model indices
    model_indices = self.m_file_selector.model_indices()

#### compute full score matrix as well
#    sim_matrix = numpy.ndarray((len(model_indices),len(model_indices)), dtype=numpy.float64)
####    

    f = open(self.m_file_selector.result_file(), 'w')
    # Concatenates the scores
    for model_index in model_indices:
      res_file = open(self.m_file_selector.model_score_file(model_index), 'r')
      f.write(res_file.read())
      res_file.close()
####
#      res_file = open(self.m_file_selector.model_score_file(model_index), 'r')
#      probe_index = 0
#      for line in res_file:
#        score = line.split(" ")[3]
#        sim_matrix[probe_index,model_index] = score
#        probe_index += 1
####
    f.close()

#### write similarity matrix
#    beemat = pyvision.BEEDistanceMatrix(sim_matrix,"","",None,is_distance=False)
#    beemat.save("/scratch/mguenther/GBU/Good/sim_matrix.txt")
####
