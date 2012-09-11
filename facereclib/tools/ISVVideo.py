#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Roy Wallace <Roy.Wallace@idiap.ch>

import bob
import numpy
import types
from . import UBMGMMVideoTool, ISVTool
from .. import utils

# Parent classes:
# - Warning: This class uses multiple inheritance! (Note: Python's resolution rule is: depth-first, left-to-right)
# - ISVTool extends UBMGMMTool by providing some additional methods for training the session variability subspace, etc.
# - UBMGMMVideoTool extends UBMGMMTool to support UBM training/enrollment/testing with video.FrameContainers
#
# Here we extend the parent classes by overriding methods:
# -- read_feature --> overridden (use UBMGMMVideoTool's, to read a video.FrameContainer)
# -- train_projector --> overridden (use UBMGMMVideoTool's)
# -- train_enroller --> overridden (based on ISVTool's, but projects only selected frames)
# -- project --> overridden (use UBMGMMVideoTool's)
# -- enroll --> overridden (based on ISVTool, but first projects only selected frames)
# -- read_model --> inherited from ISVTool (because it's inherited first)
# -- read_probe --> inherited from ISVTool (because it's inherited first)
# -- score --> inherited from ISVTool (because it's inherited first)

class ISVVideoTool (ISVTool, UBMGMMVideoTool):
  """Tool chain for video-to-video face recognition using inter-session variability modelling (ISV)."""

  def __init__(self, setup):
    ISVTool.__init__(self, setup)
    self.use_unprojected_features_for_model_enroll = True

  # Overrides ISVTool.train_enroller
  def train_enroller(self, train_files, enroller_file):
    print "-> ISVVideoTool.train_enroller"
    ########## (same as ISVTool.train_enroller)
    # create a JFABasemachine with the UBM from the base class
    self.m_jfabase = bob.machine.JFABaseMachine(self.m_ubm, self.m_config.SUBSPACE_DIMENSION_OF_U)
    self.m_jfabase.ubm = self.m_ubm

    ########## calculate GMM stats from video.FrameContainers, using frame_selector_for_train_enroller
    gmm_stats = []
    for k in sorted(train_files.keys()): # loop over clients
      gmm_stats_client = []
      for j in sorted(train_files[k].keys()): # loop over videos of client k
        frame_container = utils.video.FrameContainer(str(train_files[k][j]))
        this_gmm_stats = UBMGMMVideoTool.project(self,frame_container,self.m_config.frame_selector_for_enroller_training)
        gmm_stats_client.append(this_gmm_stats)
      gmm_stats.append(gmm_stats_client)
    print "--> got gmm_stats for " + str(len(gmm_stats)) + " clients"

    ########## (same as ISVTool.train_enroller)
    t = bob.trainer.JFABaseTrainer(self.m_jfabase)
    t.train_isv(gmm_stats, self.m_config.JFA_TRAINING_ITERATIONS, self.m_config.RELEVANCE_FACTOR)

    # Save the JFA base AND the UBM into the same file
    self.m_jfabase.save(bob.io.HDF5File(enroller_file, "w"))

  def enroll(self, frame_containers):
    print "-> ISVVideoTool.enroll"
    enroll_features = []
    for frame_container in frame_containers:
      this_enroll_features = UBMGMMVideoTool.project(self,frame_container,self.m_config.frame_selector_for_enroll)
      enroll_features.append(this_enroll_features)
    print "--> got " + str(len(enroll_features)) + " enroll_features"

    ########## (same as ISVTool.enroll)
    self.m_trainer.enroll(enroll_features, self.m_config.JFA_ENROLL_ITERATIONS)
    return self.m_machine

  def read_feature(self, feature_file):
    return UBMGMMVideoTool.read_feature(self,str(feature_file))

  def project(self, frame_container):
    """Computes GMM statistics against a UBM, given an input video.FrameContainer"""
    return UBMGMMVideoTool.project(self,frame_container)

  def train_projector(self, train_files, projector_file):
    """Computes the Universal Background Model from the training ("world") data"""
    return UBMGMMVideoTool.train_projector(self,train_files, projector_file)

