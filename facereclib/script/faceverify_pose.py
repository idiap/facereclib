#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>


import sys, os
import argparse, imp

from . import ToolChainExecutor
from .. import toolchain
from .. import utils

class ToolChainExecutorPose (ToolChainExecutor.ToolChainExecutor):
  """Class that executes the ZT tool chain (locally or in the grid)"""

  def __init__(self, args):
    utils.error("This class is deprecated.")
    # DO NOT call base class constructor
#    ToolChainExecutor.ToolChainExecutor.__init__(self, args)

    # remember command line arguments
    self.m_args = args
    # load configuration files specified on command line
    self.m_tool_config = imp.load_source('tool_chain', args.tool)
    self.m_preprocessor_config =  imp.load_source('preprocessor', args.preprocessor)
    self.m_extractor_config = imp.load_source('extractor', args.features)
    if args.grid:
      self.m_grid_config = imp.load_source('grid', args.grid)

    # generate parametrization based on protocol
    self.m_preprocessor = self.m_preprocessor_config.preprocessor(self.m_preprocessor_config)

    self.m_types = set()
    self.m_database_configs = { 'neutral' : imp.load_source('neutral', args.database) }
    self.__generate_configuration__('neutral')
    self.m_extractors = { 'neutral' : self.m_extractor_config.feature_extractor(self.m_extractor_config) }
    self.m_tools = {'neutral' : self.m_tool_config.tool(self.m_tool_config)}
    self.m_file_selectors = {'neutral' : toolchain.FileSelectorZT(self.m_database_configs['neutral'], self.m_database_configs['neutral'])}
    self.m_tool_chains = { 'neutral' : toolchain.ToolChainZT(self.m_file_selectors['neutral']) }

    def set_options(type, configuration):
      def set_single(type, configuration, varname):
        if not hasattr(configuration, varname):
          configuration.__dict__[varname] = configuration.keywords[type].items()
        else:
          for k,v in configuration.keywords[type].items():
            if k in configuration.__dict__[varname].keys():
              configuration.__dict__[varname][k].extend(v[:])
            else:
              configuration.__dict__[varname][k] = v[:]

      set_single(type, configuration, 'all_files_options')
      set_single(type, configuration, 'extractor_training_options')
      set_single(type, configuration, 'projector_training_options')
      set_single(type, configuration, 'enroller_training_options')

    for p in args.protocols:
      self.m_database_configs[p] = imp.load_source(p, args.database)
      set_options(p, self.m_database_configs[p])
      if args.train_on_specific_protocols == 'together':
        # also set the neutral conditions, if training should be done with both
        set_options('neutral', self.m_database_configs[p])
        set_options(p, self.m_database_configs['neutral'])
      self.__generate_configuration__(p)
      self.m_extractors[p] = self.m_extractor_config.feature_extractor(self.m_extractor_config)
      self.m_tools[p] = self.m_tool_config.tool(self.m_tool_config)
      self.m_file_selectors[p] = toolchain.FileSelectorZT(self.m_database_configs[p], self.m_database_configs[p])
      self.m_tool_chains[p] = toolchain.ToolChainZT(self.m_file_selectors[p])

      if args.train_on_specific_protocols != None:
        # set options for frontal
        self.m_types.add(p)

    # if we have specified protocols, add the
    if args.train_on_specific_protocols != None:
      set_options('neutral', self.m_database_configs['neutral'])

    if args.train_on_specific_protocols != 'together':
      self.m_types.add('neutral')

    if args.compute_rank_list:
      # TODO: Fill this
      pass

    utils.set_verbosity_level(args.verbose)


  def __generate_configuration__(self, type):
    configuration = self.m_database_configs[type]

    user_name = os.environ['USER']
    if self.m_args.user_dir:
      configuration.base_output_USER_dir = os.path.join(self.m_args.user_dir, self.m_args.sub_dir)
    else:
      configuration.base_output_USER_dir = os.path.join("/idiap/user", user_name, configuration.name, self.m_args.sub_dir)

    if self.m_args.temp_dir:
      configuration.base_output_TEMP_dir = os.path.join(self.m_args.temp_dir, self.m_args.sub_dir)
    else:
      if not self.m_args.grid:
        configuration.base_output_TEMP_dir = os.path.join("/scratch", user_name, configuration.name, self.m_args.sub_dir)
      else:
        configuration.base_output_TEMP_dir = os.path.join("/idiap/temp", user_name, configuration.name, self.m_args.sub_dir)

    configuration.extractor_file = os.path.join(configuration.base_output_TEMP_dir, type, self.m_args.extractor_file)
    configuration.projector_file = os.path.join(configuration.base_output_TEMP_dir, type, self.m_args.projector_file)
    configuration.enroller_file = os.path.join(configuration.base_output_TEMP_dir, type, self.m_args.enroller_file)

    configuration.preprocessed_dir = os.path.join(configuration.base_output_TEMP_dir, self.m_args.preprocessed_dir)
    configuration.features_dir = os.path.join(configuration.base_output_TEMP_dir, type, self.m_args.features_dir)
    configuration.projected_dir = os.path.join(configuration.base_output_TEMP_dir, type, self.m_args.projected_dir)
    if self.m_args.train_on_specific_protocols == 'together':
      configuration.models_dir = os.path.join(configuration.base_output_TEMP_dir, type, self.m_args.models_dirs[0])
      configuration.tnorm_models_dir = os.path.join(configuration.base_output_TEMP_dir, type, self.m_args.models_dirs[1])
    else:
      configuration.models_dir = os.path.join(configuration.base_output_TEMP_dir, configuration.protocol, self.m_args.models_dirs[0])
      configuration.tnorm_models_dir = os.path.join(configuration.base_output_TEMP_dir, configuration.protocol, self.m_args.models_dirs[1])

    if not hasattr(configuration, "default_extension"):
      configuration.default_extension = ".hdf5"
    if not hasattr(configuration, "annotation_extension"):
      configuration.annotation_extension = ".pos"

    self.set_protocol(type, type)

  def set_protocol(self, type, protocol):
    configuration = self.m_database_configs[type]

    if protocol in self.m_args.protocols:
      configuration.protocol = protocol

    configuration.zt_norm_A_dir = os.path.join(configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, protocol, self.m_args.zt_dirs[0])
    configuration.zt_norm_B_dir = os.path.join(configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, protocol, self.m_args.zt_dirs[1])
    configuration.zt_norm_C_dir = os.path.join(configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, protocol, self.m_args.zt_dirs[2])
    configuration.zt_norm_D_dir = os.path.join(configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, protocol, self.m_args.zt_dirs[3])
    configuration.zt_norm_D_sameValue_dir = os.path.join(configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, protocol, self.m_args.zt_dirs[4])

    configuration.scores_nonorm_dir = os.path.join(configuration.base_output_USER_dir, self.m_args.score_sub_dir, protocol, self.m_args.score_dirs[0])
    configuration.scores_ztnorm_dir = os.path.join(configuration.base_output_USER_dir, self.m_args.score_sub_dir, protocol, self.m_args.score_dirs[1])




  def execute_tool_chain(self):
    """Executes the ZT tool chain on the local machine"""

    # preprocessing
    if not self.m_args.skip_preprocessing:
      if self.m_args.dry_run:
        print "Would have preprocessed images ..."
      else:
        self.m_tool_chains['neutral'].preprocess_images(
              self.m_preprocessor,
              force = self.m_args.force)

    # feature extraction
    if not self.m_args.skip_extractor_training:
      for type in self.m_types:
        if hasattr(self.m_extractors[type], 'train'):
          if self.m_args.dry_run:
            print "Would have trained the extractor of type %s ..." % type
          else:
            self.m_tool_chains[type].train_extractor(
                  self.m_extractors[type],
                  self.m_preprocessor,
                  force = self.m_args.force)

    if not self.m_args.skip_extraction:
      for type in self.m_types:
        if self.m_args.dry_run:
          print "Would have extracted the features of type %s ..." % type
        else:
          self.m_tool_chains[type].extract_features(
                self.m_extractors[type],
                self.m_preprocessor,
                force = self.m_args.force)

    # feature projection
    if not self.m_args.skip_projector_training:
      for type in self.m_types:
        if self.m_args.dry_run:
          print "Would have trained the projector of type %s ..." % type
        else:
          if hasattr(self.m_tools[type], 'train_projector'):
            self.m_tool_chains[type].train_projector(
                  self.m_tools[type],
                self.m_extractors[type],
                  force = self.m_args.force)

    if not self.m_args.skip_projection:
      for type in self.m_types:
        if self.m_args.dry_run:
          print "Would have projected the features of type %s ..." % type
        else:
          if hasattr(self.m_tools[type], 'project'):
            self.m_tool_chains[type].project_features(
                  self.m_tools[type],
                  self.m_extractors[type],
                  force = self.m_args.force)

    # model enrollment:
    # if we have trained with neutral and protocol-specific types, we have to enroll for each protocol separately
    # else we can enroll only with the neutral faces only
    types = self.m_types if self.m_args.train_on_specific_protocols == 'together' else ['neutral']
    for type in types:
      if not self.m_args.skip_enroller_training and hasattr(self.m_tools[type], 'train_enroller'):
        if self.m_args.dry_run:
          print "Would have trained the enroller of type %s ..." % type
        else:
          self.m_tool_chains[type].train_enroller(
                self.m_tools[type],
                self.m_extractors[type],
                force = self.m_args.force)

      if not self.m_args.skip_enrollment:
        if self.m_args.dry_run:
          print "Would have enrolled the models of groups %s for type %s ..." % (self.m_args.groups, type)
        else:
          self.m_tool_chains[type].enroll_models(
                self.m_tools[type],
                self.m_extractors[type],
                self.m_args.zt_norm,
                groups = self.m_args.groups,
                force = self.m_args.force)

    # score computation
    if not self.m_args.skip_score_computation:
      for p in self.m_args.protocols:
        if p in self.m_types:
          type = p
        else:
          type = 'neutral'
          self.set_protocol('neutral', p)

        if self.m_args.dry_run:
          print "Would have computed the scores of groups %s for protocol %s ..." % (self.m_args.groups, p)
        else:
          self.m_tool_chains[type].compute_scores(
                self.m_tools[type],
                self.m_args.zt_norm,
                groups = self.m_args.groups,
                preload_probes = self.m_args.preload_probes,
                force = self.m_args.force)

        if self.m_args.zt_norm:
          if self.m_args.dry_run:
            print "Would have computed the ZT-norm scores of groups %s for protocol %s ..." % (self.m_args.groups, p)
          else:
            self.m_tool_chains[type].zt_norm(groups = self.m_args.groups)

        # concatenation of scores
        if not self.m_args.skip_concatenation:
          if self.m_args.dry_run:
            print "Would have concatenated the scores of groups %s for protocol %s ..." % (self.m_args.groups, p)
          else:
            self.m_tool_chains[type].concatenate(
                  self.m_args.zt_norm,
                  groups = self.m_args.groups)


  def add_jobs_to_grid(self, external_dependencies):
    """Adds all (desired) jobs of the tool chain to the grid"""

    def type_short(type):
      if type == 'neutral':
        return 'all'
      else:
        return type[1:]
    # collect the job ids
    job_ids = {}

    # if there are any external dependencies, we need to respect them
    deps = {}
    for type in self.m_types:
      deps[type] = external_dependencies[:]

    # image preprocessing; never has any dependencies.
    if not self.m_args.skip_preprocessing:
      job_id = self.submit_grid_job(
              'preprocess',
              list_to_split = self.m_file_selectors['neutral'].original_image_list(),
              number_of_files_per_job = self.m_grid_config.number_of_images_per_job,
              dependencies = [],
              **self.m_grid_config.preprocessing_queue)
      job_ids['preprocessing'] = [job_id]
      for type in self.m_types:
        deps[type].append(job_id)

    # feature extraction training
    if not self.m_args.skip_extractor_training:
      for type in self.m_types:
        if hasattr(self.m_extractors[type], 'train'):
          job_id = self.submit_grid_job(
              'train-extractor --protocol %s'%type,
              name = '%s-f-train'%type_short(type),
              dependencies = deps[type],
              **self.m_grid_config.training_queue)
          deps[type].append(job_id)
          job_ids['%s_extraction_training'] = job_id

    # feature extraction
    if not self.m_args.skip_extraction:
      for type in self.m_types:
        job_id = self.submit_grid_job(
            'extract --protocol %s'%type,
            name="%s-extract"%type_short(type),
            list_to_split = self.m_file_selectors[type].preprocessed_image_list(),
            number_of_files_per_job = self.m_grid_config.number_of_features_per_job,
            dependencies = deps[type],
            **self.m_grid_config.extraction_queue)
        deps[type].append(job_id)
        job_ids['%s_feature_extraction'%type] = job_id


    # feature projection training
    if not self.m_args.skip_projector_training:
      for type in self.m_types:
        if hasattr(self.m_tools[type], 'train_projector'):
          job_id = self.submit_grid_job(
                  'train-projector --protocol %s'%type,
                  name="%s-p-train"%type_short(type),
                  dependencies = deps[type],
                  **self.m_grid_config.training_queue)
          deps[type].append(job_id)
          job_ids['%s_projector_training'%type] = job_id

    # feature projection
    if not self.m_args.skip_projection:
      for type in self.m_types:
        if hasattr(self.m_tools[type], 'project'):
          job_id = self.submit_grid_job(
                  'project --protocol %s'%type,
                  name="%s-project"%type_short(type),
                  list_to_split = self.m_file_selectors[type].feature_list(),
                  number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
                  dependencies = deps[type],
                  **self.m_grid_config.projection_queue)
          deps[type].append(job_id)
          job_ids['%s_feature_projection'%type] = job_id

    # model enrollment training
    if not self.m_args.skip_enroller_training:
      job_ids_et = []
      for type in self.m_types:
        if hasattr(self.m_tools[type], 'train_enroller'):
          job_id = self.submit_grid_job(
                  'train-enroller',
                  name = "%s-e-train"%type_short(type),
                  dependencies = deps[type],
                  **self.m_grid_config.training_queue)
          deps[type].append(job_id)
          job_ids['%s_enrollment_training'%type] = job_id


    # model enrollment
    enroll_deps_n = {}
    enroll_deps_t = {}
    for group in self.m_args.groups:
      enroll_deps_n[group] = {'neutral':[]}
      enroll_deps_t[group] = {'neutral':[]}
      for p in self.m_args.protocols:
        enroll_deps_n[group][p] = deps[type][:]
        enroll_deps_t[group][p] = deps[type][:]

    for group in self.m_args.groups:
      if not self.m_args.skip_enrollment:
        for type in self.m_types:
          job_id = self.submit_grid_job(
                  'enroll --protocol %s --group %s --model-type N'%(type, group),
                  name = "%s-en-N-%s"%(type_short(type),group),
                  list_to_split = self.m_file_selectors[type].model_ids(group),
                  number_of_files_per_job = self.m_grid_config.number_of_models_per_enroll_job,
                  dependencies = deps[type],
                  **self.m_grid_config.enroll_queue)
          enroll_deps_n[group][type].append(job_id)
          job_ids['%s_enroll_%s_N'%(type,group)] = job_id

          if self.m_args.zt_norm:
            job_id = self.submit_grid_job(
                    'enroll --protocol %s --group %s --model-type T'%(type,group),
                    name = "%s-en-T-%s"%(type_short(type),group),
                    list_to_split = self.m_file_selectors[type].tmodel_ids(group),
                    number_of_files_per_job = self.m_grid_config.number_of_models_per_enroll_job,
                    dependencies = deps[type],
                    **self.m_grid_config.enroll_queue)
            enroll_deps_t[group][type].append(job_id)
            job_ids['%s_enroll_%s_T'%(type,group)] = job_id

      # compute A, B, C, and D scores
      score_deps = {}
      concat_deps = {}
      if not self.m_args.skip_score_computation:
        for p in self.m_args.protocols:
          job_id = self.submit_grid_job(
                  'compute-scores --protocol %s --group %s --score-type A'%(p,group),
                  name = "%s-sc-A-%s"%(type_short(p),group),
                  list_to_split = self.m_file_selectors[p].model_ids(group),
                  number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job,
                  dependencies = enroll_deps_n[group][p] + enroll_deps_n[group]['neutral'],
                  **self.m_grid_config.score_queue)

          job_ids['%s_score_%s_A'%(p,group)]  = job_id
          concat_deps[p] = [job_id]
          score_deps[p] = [job_id]

          if self.m_args.zt_norm:
            job_id = self.submit_grid_job(
                    'compute-scores --protocol %s --group %s --score-type B'%(p,group),
                    name = "%s-sc-B-%s"%(type_short(p),group),
                    list_to_split = self.m_file_selectors[p].model_ids(group),
                    number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job,
                    dependencies = enroll_deps_n[group][p] + enroll_deps_n[group]['neutral'],
                    **self.m_grid_config.score_queue)

            job_ids['%s_score_%s_B'%(p,group)] = job_id
            score_deps[p].append(job_id)

            for m in ('C', 'D'):
              job_id = self.submit_grid_job(
                    'compute-scores --protocol %s --group %s --score-type %s'%(p,group,m),
                    name = "%s-sc-%s-%s"%(type_short(p),m,group),
                    list_to_split = self.m_file_selectors[p].tmodel_ids(group),
                    number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job,
                    dependencies = enroll_deps_t[group][p] + enroll_deps_t[group]['neutral'],
                    **self.m_grid_config.score_queue)

              job_ids['%s_sc_%s_%s'%(p,group,m)] = job_id
              score_deps[p].append(job_id)

            # compute zt-norm
            job_id = self.submit_grid_job(
                    'compute-scores --protocol %s --group %s --score-type Z'%(p,group),
                    name = "%s-sc-Z-%s"%(type_short(p),group),
                    dependencies = score_deps[p])
            concat_deps[p].append(job_id)
            job_ids['%s_score_%s_Z'%(p,group)] = job_id
      else:
        concat_deps[p] = []

      # concatenate results
      if not self.m_args.skip_concatenation:
        for p in self.m_args.protocols:
          job_id = self.submit_grid_job(
                  'concatenate --protocol %s --group %s'%(p,group),
                  name = "%s-con-%s"%(type_short(p),group),
                  dependencies = concat_deps[p])
          job_ids['%s_concat_%s'%(p,group)] = job_id

    # return the job ids, in case anyone wants to know them
    return job_ids


  def execute_grid_job(self):
    """Run the desired job of the ZT tool chain that is specified on command line"""
    # preprocess
    if self.m_args.sub_task == 'preprocess':
      self.m_tool_chains['neutral'].preprocess_images(
          self.m_preprocessor,
          indices = self.indices(self.m_file_selectors['neutral'].original_image_list(), self.m_grid_config.number_of_images_per_job),
          force = self.m_args.force)

    # feature extraction training
    elif self.m_args.sub_task == 'train-extractor':
      self.m_tool_chains[self.m_args.protocol].train_extractor(
          self.m_extractors[self.m_args.protocol],
          self.m_preprocessor,
          force = self.m_args.force)

    # extract features
    elif self.m_args.sub_task == 'extract':
      self.m_tool_chains[self.m_args.protocol].extract_features(
          self.m_extractors[self.m_args.protocol],
          self.m_preprocessor,
          indices = self.indices(self.m_file_selectors[self.m_args.protocol].preprocessed_image_list(), self.m_grid_config.number_of_features_per_job),
          force = self.m_args.force)

    # train the feature projector
    elif self.m_args.sub_task == 'train-projector':
      self.m_tool_chains[self.m_args.protocol].train_projector(
          self.m_tools[self.m_args.protocol],
          self.m_extractors[self.m_args.protocol],
          force = self.m_args.force)

    # project the features
    elif self.m_args.sub_task == 'project':
      self.m_tool_chains[self.m_args.protocol].project_features(
          self.m_tools[self.m_args.protocol],
          self.m_extractors[self.m_args.protocol],
          indices = self.indices(self.m_file_selectors[self.m_args.protocol].preprocessed_image_list(), self.m_grid_config.number_of_projections_per_job),
          force = self.m_args.force)

    # train model enroller
    elif self.m_args.sub_task == 'train-enroller':
      self.m_tool_chains[self.m_args.protocol].train_enroller(
          self.m_tools[self.m_args.protocol],
          self.m_extractors[self.m_args.protocol],
          force = self.m_args.force)

    # enroll models
    elif self.m_args.sub_task == 'enroll':
      if self.m_args.model_type == 'N':
        self.m_tool_chains[self.m_args.protocol].enroll_models(
            self.m_tools[self.m_args.protocol],
            self.m_extractors[self.m_args.protocol],
            self.m_args.zt_norm,
            indices = self.indices(self.m_file_selectors[self.m_args.protocol].model_ids(self.m_args.group), self.m_grid_config.number_of_models_per_enroll_job),
            groups = [self.m_args.group],
            types = ['N'],
            force = self.m_args.force)

      else:
        self.m_tool_chains[self.m_args.protocol].enroll_models(
            self.m_tools[self.m_args.protocol],
            self.m_extractors[self.m_args.protocol],
            self.m_args.zt_norm,
            indices = self.indices(self.m_file_selectors[self.m_args.protocol].tmodel_ids(self.m_args.group), self.m_grid_config.number_of_models_per_enroll_job),
            groups = [self.m_args.group],
            types = ['T'],
            force = self.m_args.force)

    # compute scores
    elif self.m_args.sub_task == 'compute-scores':
      if self.m_args.protocol in self.m_types:
        type = self.m_args.protocol
      else:
        type = 'neutral'
        self.set_protocol('neutral', self.m_args.protocol)

      if self.m_args.score_type in ['A', 'B']:
        self.m_tool_chains[type].compute_scores(
            self.m_tools[type],
            self.m_args.zt_norm,
            indices = self.indices(self.m_file_selectors[type].model_ids(self.m_args.group), self.m_grid_config.number_of_models_per_score_job),
            groups = [self.m_args.group],
            types = [self.m_args.score_type],
            preload_probes = self.m_args.preload_probes,
            force = self.m_args.force)

      elif self.m_args.score_type in ['C', 'D']:
        self.m_tool_chains[type].compute_scores(
            self.m_tools[type],
            self.m_args.zt_norm,
            indices = self.indices(self.m_file_selectors[type].tmodel_ids(self.m_args.group), self.m_grid_config.number_of_models_per_score_job),
            groups = [self.m_args.group],
            types = [self.m_args.score_type],
            preload_probes = self.m_args.preload_probes,
            force = self.m_args.force)

      else:
        self.m_tool_chains[type].zt_norm(groups = [self.m_args.group])

    # concatenate
    elif self.m_args.sub_task == 'concatenate':
      if self.m_args.protocol in self.m_types:
        type = self.m_args.protocol
      else:
        type = 'neutral'
        self.set_protocol('neutral', self.m_args.protocol)

      self.m_tool_chains[type].concatenate(
          self.m_args.zt_norm,
          groups = [self.m_args.group])

    # Test if the keyword was processed
    else:
      raise ValueError("The given subtask '%s' could not be processed. THIS IS A BUG. Please report this to the authors.")

def parse_args(command_line_arguments = sys.argv[1:]):
  """This function parses the given options (which by default are the command line options)"""
  # sorry for that.
  global parameters
  parameters = command_line_arguments

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # add the arguments required for all tool chains
  config_group, dir_group, file_group, sub_dir_group, other_group, skip_group = ToolChainExecutorPose.required_command_line_options(parser)

  config_group.add_argument('-P', '--protocols', type = str, nargs = '+', required = True,
      help = 'The protocols that should be evaluated')

  sub_dir_group.add_argument('--models-directories', type = str, metavar = 'DIR', nargs = 2, dest='models_dirs',
      default = ['models', 'tmodels'],
      help = 'Sub-directories (of --temp-directory) where the models should be stored')
  sub_dir_group.add_argument('--zt-norm-directories', type = str, metavar = 'DIR', nargs = 5, dest='zt_dirs',
      default = ['zt_norm_A', 'zt_norm_B', 'zt_norm_C', 'zt_norm_D', 'zt_norm_D_sameValue'],
      help = 'Sub-directories (of --temp-directory) where to write the zt_norm values')
  sub_dir_group.add_argument('--zt-score-directories', type = str, metavar = 'DIR', nargs = 2, dest='score_dirs',
      default = ['nonorm', 'ztnorm'],
      help = 'Subdirectories (of --user-dir) where to write the results to')

  #######################################################################################
  ############################ other options ############################################
  other_group.add_argument('-z', '--zt-norm', action='store_true',
      help = 'Enables the computation of ZT norms')
  other_group.add_argument('-X', '--train-on-specific-protocols', choices = ('together', 'separate'),
      help = 'Train on data that are specific to the protocol')
  other_group.add_argument('-r', '--compute-rank-list', action = 'store_true',
      help = 'Use rank list computation to score')
  other_group.add_argument('-F', '--force', action='store_true',
      help = 'Force to erase former data if already exist')
  other_group.add_argument('-w', '--preload-probes', action='store_true',
      help = 'Preload probe files during score computation (needs more memory, but is faster and requires fewer file accesses). WARNING! Use this flag with care!')
  other_group.add_argument('--groups', type = str,  metavar = 'GROUP', nargs = '+', default = ['dev'],
      help = "The group (i.e., 'dev' or  'eval') for which the models and scores should be generated")

  #######################################################################################
  #################### sub-tasks being executed by this script ##########################
  parser.add_argument('--sub-task',
      choices = ('preprocess', 'train-extractor', 'extract', 'train-projector', 'project', 'train-enroller', 'enroll', 'compute-scores', 'concatenate'),
      help = argparse.SUPPRESS) #'Executes a subtask (FOR INTERNAL USE ONLY!!!)'
  parser.add_argument('--protocol', type = str,
      help = argparse.SUPPRESS) #'The protocol to execute'
  parser.add_argument('--model-type', type = str, choices = ['N', 'T'], metavar = 'TYPE',
      help = argparse.SUPPRESS) #'Which type of models to generate (Normal or TModels)'
  parser.add_argument('--score-type', type = str, choices=['A', 'B', 'C', 'D', 'Z'],  metavar = 'SCORE',
      help = argparse.SUPPRESS) #'The type of scores that should be computed'
  parser.add_argument('--group', type = str,  metavar = 'GROUP',
      help = argparse.SUPPRESS) #'The group for which the current action should be performed'

  return parser.parse_args(command_line_arguments)


def face_verify(args, external_dependencies = [], external_fake_job_id = 0):
  """This is the main entry point for computing face verification experiments.
  You just have to specify configuration scripts for any of the steps of the toolchain, which are:
  -- the database
  -- feature extraction (including image preprocessing)
  -- the score computation tool
  -- and the grid configuration (in case, the function should be executed in the grid).
  Additionally, you can skip parts of the toolchain by selecting proper --skip-... parameters.
  If your probe files are not too big, you can also specify the --preload-probes switch to speed up the score computation.
  If files should be re-generated, please specify the --force option (might be combined with the --skip-... options)"""


  # generate tool chain executor
  executor = ToolChainExecutorPose(args)
  # as the main entry point, check whether the grid option was given
  if not args.grid:
    # not in a grid, use default tool chain sequentially
    executor.execute_tool_chain()
    return []

  elif args.sub_task:
    # execute the desired sub-task
    executor.execute_grid_job()
    return []
  else:
    # no other parameter given, so deploy new jobs

    # get the name of this file
    this_file = __file__
    if this_file[-1] == 'c':
      this_file = this_file[0:-1]

    # initialize the executor to submit the jobs to the grid
    global parameters
    executor.set_common_parameters(calling_file = this_file, parameters = parameters, fake_job_id = external_fake_job_id, temp_dir = executor.m_database_configs['neutral'].base_output_TEMP_dir )

    # add the jobs
    return executor.add_jobs_to_grid(external_dependencies)


def main():
  """Executes the main function"""
  # do the command line parsing
  args = parse_args()
  # verify that the input files exist
  for f in (args.database, args.preprocessor, args.features, args.tool):
    if not os.path.exists(str(f)):
      raise ValueError("The given file '%s' does not exist."%f)
  # perform face verification test
  face_verify(args)

if __name__ == "__main__":
  main()

