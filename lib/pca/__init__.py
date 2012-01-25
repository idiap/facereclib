#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import os
import bob
import numpy as np

def pca_train(world_features, output_machine, n_outputs):
  """Generates the PCA covariance matrix"""
  # Initializes an arrayset for the data
  data = bob.io.Arrayset()
  for k in sorted(world_features.keys()):
    # Loads the file
    img = bob.io.load( str(world_features[k]) )
    # Appends in the arrayset
    data.append(img)

  print "Training LinearMachine using PCA (SVD)"
  T = bob.trainer.SVDPCATrainer()
  machine, eig_vals = T.train(data)
  # Machine: get shape, then resize
  machine.resize(machine.shape[0], n_outputs)
  machine.save(bob.io.HDF5File(output_machine))


def pca_project(input_features, input_machine, output_features, force=False):
  """Projects the data using the provided covariance matrix"""
  # Loads the machine (linear projection matrix)
  if not os.path.exists(input_machine):
      raise RuntimeError, "Cannot find Linear PCA Machine %s" % (input_machine)
  machine = bob.machine.LinearMachine(bob.io.HDF5File(input_machine))

  # Allocates an array for the projected data
  img_out = np.ndarray(machine.shape[1], 'float64')

  for k in sorted(input_features.keys()):
    if force == True and os.path.exists(output_features[k]):
      print "Removing old features %s." % (output_features[k])
      os.remove(output_features[k])

    if os.path.exists(output_features[k]):
      print "Projected features %s already exists."  % (output_features[k])
    else:
      print "Computing projected features from sample %s." % (input_features[k])
      # Loads the data
      img_in = bob.io.load( str(input_features[k]) )
      # Projects the data
      machine(img_in, img_out)
      # Saves the projected data
      bob.io.save(img_out, str(output_features[k]))


def pca_enrol_model(enrol_files, model_path):
  """Enrols a client model and saves it to the given file"""
  c = 0 # counts the number of enrolment files
  model = np.ndarray(0, 'float64')
  for k in sorted(enrol_files.keys()):
    # Processes one file
    img = bob.io.load( str(enrol_files[k]) )

    if model.shape[0] == 0:
      model.resize(img.shape[0])
      model[:]=0
    if img.shape[0] != model.shape[0]:
      raise Exception('Size mismatched')

    model += img
    c += 1

  # Normalizes the model
  model /= c
  # Saves to file
  bob.io.save(model,model_path)


def pca_scores(model_file, probe_files):
  """Computes the score of each probe against the model. 
     The measure is the negative Euclidean distance.
     A high score (negative value close) to zero occurs when a probe 
     matches the model"""
  scores = np.ndarray((1,len(probe_files)), 'float64')
  # Loads the model
  if not os.path.exists(model_file):
    raise RuntimeError, "Could not find model %s." % model_file
  model = bob.io.load( str(model_file) )
  # Loops over the probes
  i = 0
  for f in probe_files:
    if not os.path.exists(str(f)):
      raise RuntimeError, "Could not find model %s." % str(f)
    probe = bob.io.load(str(f))
    scores[0,i] = -np.linalg.norm(probe-model)
    i += 1
  # Returns the (negative) distance between the samples and the model
  return scores


def pca_scores_A(models_ids, models_dir, probe_files, db,
                 zt_norm_A_dir, scores_nonorm_dir, group, probes_split_id):
  """Computes a split of the A matrix for the ZT-Norm and saves the raw scores to file"""
  
  # Gets the probe samples
  probe_tests = []
  for k in sorted(probe_files.keys()):
    probe_tests.append(str(probe_files[k][0]))

  # Computes the raw scores for each model
  for model_id in models_ids:
    model_path = os.path.join(models_dir, str(model_id) + ".hdf5")
    A = pca_scores(model_path, probe_tests)
    bob.io.save(A,os.path.join(zt_norm_A_dir, group, str(model_id) + "_" + str(probes_split_id).zfill(4) + ".hdf5"))

    # Saves to text file
    import utils
    scores_list = utils.convertScoreToList(np.reshape(A,A.size), probe_files)
    sc_nonorm_filename = os.path.join(scores_nonorm_dir, group, str(model_id) + "_" + str(probes_split_id).zfill(4) + ".txt")
    f_nonorm = open(sc_nonorm_filename, 'w')
    for x in scores_list:
      f_nonorm.write(str(x[2]) + " " + str(x[0]) + " " + str(x[3]) + " " + str(x[4]) + "\n")
    f_nonorm.close()


def pca_ztnorm_B(models_ids, models_dir, zfiles, db,
                 zt_norm_B_dir, group, zsamples_split_id):
  """Computes a split of the B matrix for the ZT-Norm"""
  
  # Gets the Z-Norm impostor samples
  zprobe_tests = []
  for k in sorted(zfiles.keys()):
    zprobe_tests.append(str(zfiles[k][0]))

  # Loads the models
  for model_id in models_ids:
    model_path = os.path.join(models_dir, str(model_id) + ".hdf5")
    B = pca_scores(model_path, zprobe_tests)
    bob.io.save(B,os.path.join(zt_norm_B_dir, group, str(model_id) + "_" + str(zsamples_split_id).zfill(4) + ".hdf5"))


def pca_ztnorm_C(tmodel_id, tnorm_models_dir, probe_files, db,
                 zt_norm_C_dir, group, probes_split_id):
  """Computes a split of the C matrix for the ZT-Norm"""

  # Gets the probe samples
  probe_tests = []
  for k in sorted(probe_files.keys()):
    probe_tests.append(str(probe_files[k][0]))

  # Computes the raw scores for the T-Norm model
  model_path = os.path.join(tnorm_models_dir, str(tmodel_id) + ".hdf5")
  C = pca_scores(model_path, probe_tests)
  bob.io.save(C,os.path.join(zt_norm_C_dir, group, "TM" + str(tmodel_id) + "_" + str(probes_split_id).zfill(4) + ".hdf5"))


def pca_ztnorm_D(tnorm_models_ids, tnorm_models_dir, zfiles, db,
                 zt_norm_D_dir, zt_norm_D_sameValue_dir, group, zsamples_split_id):
  """Computes a split of the D matrix for the ZT-Norm"""

  # Gets the Z-Norm impostor samples
  zprobe_tests = []
  znorm_clients_ids = []
  for k in sorted(zfiles.keys()):
    zprobe_tests.append(str(zfiles[k][0]))
    znorm_clients_ids.append(zfiles[k][3])

  # Loads the T-Norm models
  for tmodel_id in tnorm_models_ids:
    tmodel_path = os.path.join(tnorm_models_dir, str(tmodel_id) + ".hdf5")
    D = pca_scores(tmodel_path, zprobe_tests)
    bob.io.save(D,os.path.join(zt_norm_D_dir, group, str(tmodel_id) + "_" + str(zsamples_split_id).zfill(4) + ".hdf5"))

    tnorm_clients_ids = [db.getClientIdFromModelId(tmodel_id)]
    D_sameValue_tm = bob.machine.ztnormSameValue(tnorm_clients_ids, znorm_clients_ids)
    bob.io.save(D_sameValue_tm,os.path.join(zt_norm_D_sameValue_dir, group, str(tmodel_id) + "_" + str(zsamples_split_id).zfill(4) + ".hdf5"))
