#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import os
import utils
import bob
import numpy as np

def load_gmmstats(l_files):
  """Loads a dictionary of GMM statistics from a list of filenames"""
  gmmstats = [] 
  for k in l_files: 
    # Processes one file 
    stats = bob.machine.GMMStats( bob.io.HDF5File(str(l_files[k])) ) 
    # Appends in the list 
    gmmstats.append(stats)
  return gmmstats


def load_gmmstats_train(ld_files):
  """Loads a list of lists of GMM statistics from a list of dictionaries of filenames
     There is one list for each identity"""

  # Initializes a python list for the GMMStats
  gmmstats = [] 
  for l in ld_files: 
    # Loads the list of GMMStats for the given client
    gmmstats_c = load_gmmstats(l)
    # Appends to the main list 
    gmmstats.append(gmmstats_c)
  return gmmstats


def jfa_train_base(train_files, jfabase_filename, ubm_filename,
                  ru=2, rv=2, n_iter_train=10, force=False):
  """Trains a Universal Background Model and saves it to file"""

  # Removes old file if required
  if force == True and os.path.exists(jfabase_filename):
    print "Remove old file %s." % (jfabase_filename)
    os.remove(jfabase_filename)

  if os.path.exists(jfabase_filename):
    print "JFABase model already exists"
  else:
    print "Training JFABase model"
    if not os.path.exists(ubm_filename):
      raise RuntimeError, "Cannot find UBM %s" % (ubm_filename)
    ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_filename))

    jfabase = bob.machine.JFABaseMachine(ubm, ru, rv)
    jfabase.ubm = ubm

    gmmstats = load_gmmstats_train(train_files)

    T = bob.trainer.JFABaseTrainer(jfabase)
    T.train(gmmstats, n_iter_train)

    # Save the JFA base to file
    jfabase.save(bob.io.HDF5File(jfabase_filename))


def isv_train_base(train_files, isvbase_filename, ubm_filename,
                  ru=50, relevance_factor=4, n_iter_train=10, force=False):
  """Trains a Universal Background Model and saves it to file"""

  # Removes old file if required
  if force == True and os.path.exists(isvbase_filename):
    print "Remove old file %s." % (isvbase_filename)
    os.remove(isvbase_filename)

  if os.path.exists(isvbase_filename):
    print "ISV Base model already exists"
  else:
    print "Training ISV Base model"
    if not os.path.exists(ubm_filename):
      raise RuntimeError, "Cannot find UBM %s" % (ubm_filename)
    ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_filename))

    isvbase = bob.machine.JFABaseMachine(ubm, ru, 1)
    isvbase.ubm = ubm

    gmmstats = load_gmmstats_train(train_files)

    T = bob.trainer.JFABaseTrainer(isvbase)
    T.trainISV(gmmstats, n_iter_train, relevance_factor)

    # Save the ISV base to file
    isvbase.save(bob.io.HDF5File(isvbase_filename))


def isv_merge_base(isvbase_filename1, isvbase_filename2, ubm_filename, 
    isvbase_filename, force=False):
  # Removes old file if required
  if force == True and os.path.exists(isvbase_filename):
    print "Remove old file %s." % (isvbase_filename)
    os.remove(isvbase_filename)

  if os.path.exists(isvbase_filename):
    print "ISV Base model already exists"
  else:
    print "Merging ISV Base models"
    if not os.path.exists(ubm_filename):
      raise RuntimeError, "Cannot find UBM %s" % (ubm_filename)
    ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_filename))

    if not os.path.exists(isvbase_filename1):
      raise RuntimeError, "Cannot find isvbase1 %s" % (isvbase_filename1)
    isv1 = bob.machine.JFABaseMachine(bob.io.HDF5File(isvbase_filename1))
    ru1 = isv1.DimRu

    if not os.path.exists(isvbase_filename2):
      raise RuntimeError, "Cannot find isvbase2 %s" % (isvbase_filename2)
    isv2 = bob.machine.JFABaseMachine(bob.io.HDF5File(isvbase_filename2))
    ru2 = isv2.DimRu
    ru = ru1 + ru2    

    print "Dimensionality of ISV: %d" % (ru)
    isv = bob.machine.JFABaseMachine(ubm, ru, 1)
    isv.ubm = ubm
    isv.U[:,0:ru1] = isv1.U
    isv.U[:,ru1:ru] = isv2.U
    isv.V = isv1.V
    isv.D = isv1.D

    # Save the ISV base to file
    isv.save(bob.io.HDF5File(isvbase_filename))


def jfa_enrol_model(enrol_files, model_path, jfabase_filename, ubm_filename, n_iter_enrol=1, force=False):
  """Enrols a JFA Machine for a given identity"""
  # Removes old file if required
  if force == True and os.path.exists(jfabase_filename):
    print "Remove old file %s." % (jfabase_filename)
    os.remove(jfabase_filename)

  if os.path.exists(model_path):
    print "JFA model already exists"
  else:
    print "Training JFA model"

    # Loads the GMM statistics
    gmmstats = load_gmmstats(enrol_files)
 
    # Loads the UBM and JFA base 
    if not os.path.exists(ubm_filename):
        raise RuntimeError, "Cannot find UBM %s" % (ubm_filename)
    ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_filename))
    if not os.path.exists(jfabase_filename):
        raise RuntimeError, "Cannot find JFA Base %s" % (jfabase_filename)
    jfabase = bob.machine.JFABaseMachine(bob.io.HDF5File(jfabase_filename))
    jfabase.ubm = ubm

    # Enrols
    machine = bob.machine.JFAMachine(jfabase)
    base_trainer = bob.trainer.JFABaseTrainer(jfabase)
    trainer = bob.trainer.JFATrainer(machine, base_trainer)
    trainer.enrol(gmmstats, n_iter_enrol)

    # Saves it to the given file
    machine.save(bob.io.HDF5File(model_path))


def jfa_scores_A(models_ids, models_dir, probe_files, jfabase_filename, ubm_filename, db,
                 zt_norm_A_dir, scores_nonorm_dir, group, probes_split_id):
  """Computes a split of the A matrix for the ZT-Norm and saves the raw scores to file"""
  
  # Loads the UBM 
  if not os.path.exists(ubm_filename):
      raise RuntimeError, "Cannot find UBM %s" % (ubm_filename) 
  ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_filename))    

  # Loads the JFA base
  if not os.path.exists(jfabase_filename):
      raise RuntimeError, "Cannot find JFA Base %s" % (jfabase_filename) 
  jfabase = bob.machine.JFABaseMachine(bob.io.HDF5File(jfabase_filename))
  jfabase.ubm = ubm

  # Gets the probe samples (as well as their corresponding client ids)
  probe_tests = []
  probe_clients_ids = []
  for k in sorted(probe_files.keys()):
    if not os.path.exists(str(probe_files[k][0])):
      raise RuntimeError, "Cannot find GMM statistics %s for this Z-Norm sample." % (probe_files[k][0])
    stats = bob.machine.GMMStats(bob.io.HDF5File(str(probe_files[k][0])))
    probe_tests.append(stats)
    probe_clients_ids.append(probe_files[k][3])

  # Loads the models
  models = []
  clients_ids = []
  for model_id in models_ids:
    model_path = os.path.join(models_dir, str(model_id) + ".hdf5")
    if not os.path.exists(model_path):
      raise RuntimeError, "Could not find model %s." % model_path
    machine = bob.machine.JFAMachine(bob.io.HDF5File(model_path))
    machine.jfa_base = jfabase
    clients_ids = [db.getClientIdFromModelId(model_id)]

    # Saves the A row vector for each model and Z-Norm samples split
    A = np.ndarray((len(probe_tests),), 'float64')
    machine.forward(probe_tests, A)
    bob.io.save(np.reshape(A,(1,A.size)),os.path.join(zt_norm_A_dir, group, str(model_id) + "_" + str(probes_split_id).zfill(4) + ".hdf5"))

    # Saves to text file
    import utils
    scores_list = utils.convertScoreToList(np.reshape(A,A.size), probe_files)
    sc_nonorm_filename = os.path.join(scores_nonorm_dir, group, str(model_id) + "_" + str(probes_split_id).zfill(4) + ".txt")
    f_nonorm = open(sc_nonorm_filename, 'w')
    for x in scores_list:
      f_nonorm.write(str(x[2]) + " " + str(x[0]) + " " + str(x[3]) + " " + str(x[4]) + "\n")
    f_nonorm.close()


def jfa_ztnorm_B(models_ids, models_dir, zfiles, jfabase_filename, ubm_filename, db,
                 zt_norm_B_dir, group, zsamples_split_id):
  """Computes a split of the B matrix for the ZT-Norm"""
  
  # Loads the UBM 
  if not os.path.exists(ubm_filename):
      raise RuntimeError, "Cannot find UBM %s" % (ubm_filename) 
  ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_filename))    

  # Loads the JFA base
  if not os.path.exists(jfabase_filename):
      raise RuntimeError, "Cannot find JFA Base %s" % (jfabase_filename) 
  jfabase = bob.machine.JFABaseMachine(bob.io.HDF5File(jfabase_filename))
  jfabase.ubm = ubm

  # Gets the Z-Norm impostor samples (as well as their corresponding client ids)
  znorm_tests = []
  znorm_clients_ids = []
  for k in sorted(zfiles.keys()):
    if not os.path.exists(str(zfiles[k][0])):
      raise RuntimeError, "Cannot find GMM statistics %s for this Z-Norm sample." % (zfiles[k][0])
    stats = bob.machine.GMMStats(bob.io.HDF5File(str(zfiles[k][0])))
    znorm_tests.append(stats)
    znorm_clients_ids.append(zfiles[k][3])

  # Loads the models
  models = []
  clients_ids = []
  for model_id in models_ids:
    model_path = os.path.join(models_dir, str(model_id) + ".hdf5")
    if not os.path.exists(model_path):
      raise RuntimeError, "Could not find model %s." % model_path
    machine = bob.machine.JFAMachine(bob.io.HDF5File(model_path))
    machine.jfa_base = jfabase

    # Save the B row vector for each model and Z-Norm samples split
    B = np.ndarray((len(znorm_tests),), 'float64')
    machine.forward(znorm_tests, B) 
    bob.io.save(np.reshape(B,(1,B.size)),os.path.join(zt_norm_B_dir, group, str(model_id) + "_" + str(zsamples_split_id).zfill(4) + ".hdf5"))


def jfa_ztnorm_C(tmodel_id, tnorm_models_dir, probe_files, jfabase_filename, ubm_filename, db,
                 zt_norm_C_dir, group, probes_split_id):
  """Computes a split of the C matrix for the ZT-Norm"""
  
  # Loads the UBM 
  if not os.path.exists(ubm_filename):
      raise RuntimeError, "Cannot find UBM %s" % (ubm_filename) 
  ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_filename))    

  # Loads the JFA base
  if not os.path.exists(jfabase_filename):
      raise RuntimeError, "Cannot find JFA Base %s" % (jfabase_filename) 
  jfabase = bob.machine.JFABaseMachine(bob.io.HDF5File(jfabase_filename))
  jfabase.ubm = ubm

  # Gets the probe samples (as well as their corresponding client ids)
  probe_tests = []
  probe_clients_ids = []
  for k in sorted(probe_files.keys()):
    if not os.path.exists(str(probe_files[k])):
      raise RuntimeError, "Cannot find GMM statistics %s for this sample." % (probe_files[k])
    stats = bob.machine.GMMStats(bob.io.HDF5File(str(probe_files[k])))
    probe_tests.append(stats)

  # Loads the T-norm model
  tmodel_path = os.path.join(tnorm_models_dir, str(tmodel_id) + ".hdf5")
  if not os.path.exists(tmodel_path):
    raise RuntimeError, "Could not find T-Norm model %s." % tmodel_path
  tmachine = bob.machine.JFAMachine(bob.io.HDF5File(tmodel_path))
  tmachine.jfa_base = jfabase

  # Saves the C row vector for each T-Norm model and samples split
  C = np.ndarray((len(probe_tests),), 'float64')
  tmachine.forward(probe_tests, C) 
  bob.io.save(np.reshape(C,(1,C.size)),os.path.join(zt_norm_C_dir, group, "TM" + str(tmodel_id) + "_" + str(probes_split_id).zfill(4) + ".hdf5"))


def jfa_ztnorm_D(tnorm_models_ids, tnorm_models_dir, zfiles, jfabase_filename, ubm_filename, db,
                 zt_norm_D_dir, zt_norm_D_sameValue_dir, group, zsamples_split_id):
  """Computes a split of the D matrix for the ZT-Norm"""
  
  # Loads the UBM 
  if not os.path.exists(ubm_filename):
      raise RuntimeError, "Cannot find UBM %s" % (ubm_filename) 
  ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_filename))    

  # Loads the JFA base
  if not os.path.exists(jfabase_filename):
      raise RuntimeError, "Cannot find JFA Base %s" % (jfabase_filename) 
  jfabase = bob.machine.JFABaseMachine(bob.io.HDF5File(jfabase_filename))
  jfabase.ubm = ubm

  # Gets the Z-Norm impostor samples (as well as their corresponding client ids)
  znorm_tests = []
  znorm_clients_ids = []
  for k in sorted(zfiles.keys()):
    if not os.path.exists(str(zfiles[k][0])):
      raise RuntimeError, "Cannot find GMM statistics %s for this Z-Norm sample." % (zfiles[k][0])
    stats = bob.machine.GMMStats(bob.io.HDF5File(str(zfiles[k][0])))
    znorm_tests.append(stats)
    znorm_clients_ids.append(zfiles[k][3])

  # Loads the T-Norm models
  tnorm_models = []
  tnorm_clients_ids = []
  for tmodel_id in tnorm_models_ids:
    tmodel_path = os.path.join(tnorm_models_dir, str(tmodel_id) + ".hdf5")
    if not os.path.exists(tmodel_path):
      raise RuntimeError, "Could not find T-Norm model %s." % tmodel_path
    tmachine = bob.machine.JFAMachine(bob.io.HDF5File(tmodel_path))
    tmachine.jfa_base = jfabase
    tnorm_clients_ids = [db.getClientIdFromModelId(tmodel_id)]

    # Save the D and D_sameValue row vector for each T-Norm model and Z-Norm samples split
    D_tm = np.ndarray((len(znorm_tests),), 'float64')
    tmachine.forward(znorm_tests, D_tm) 
    bob.io.save(np.reshape(D_tm,(1,D_tm.size)), os.path.join(zt_norm_D_dir, group, str(tmodel_id) + "_" + str(zsamples_split_id).zfill(4) + ".hdf5"))
    D_sameValue_tm = bob.machine.ztnormSameValue(tnorm_clients_ids, znorm_clients_ids)
    bob.io.save(D_sameValue_tm,os.path.join(zt_norm_D_sameValue_dir, group, str(tmodel_id) + "_" + str(zsamples_split_id).zfill(4) + ".hdf5"))
