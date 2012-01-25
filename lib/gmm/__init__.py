#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import os
import utils
import bob
import numpy as np

def load_data(files):
  """Concatenates a list of blitz arrays into an Arrayset"""
  data = bob.io.Arrayset()
  for f in files:
    data.extend(bob.io.load(str(f)))
  return data


def NormalizeStdArrayset(arrayset):
  """Applies a unit variance normalization to an arrayset"""
  # Loads the data in RAM
  arrayset.load()

  # Initializes variables
  length = arrayset.shape[0]
  n_samples = len(arrayset)
  mean = np.ndarray((length,), 'float64')
  std = np.ndarray((length,), 'float64')

  mean.fill(0)
  std.fill(0)

  # Computes mean and variance
  for array in arrayset:
    x = array.astype('float64')
    mean += x
    std += (x ** 2)

  mean /= n_samples
  std /= n_samples
  std -= (mean ** 2)
  std = std ** 0.5 # sqrt(std)

  arStd = bob.io.Arrayset()
  for array in arrayset:
    arStd.append(array.astype('float64') / std)

  return (arStd,std)


def multiplyVectorsByFactors(matrix, vector):
  """Used to unnormalise some data"""
  for i in range(0, matrix.shape[0]):
    for j in range(0, matrix.shape[1]):
      matrix[i, j] *= vector[j]


def gmm_train_UBM(train_files, ubm_filename, 
                  n_gaussians=512, iterk=500, iterg=500, convergence_threshold=0.0005, variance_threshold=0.0005,
                  update_weights=True, update_means=True, update_variances=True, norm_KMeans=False):
  """Trains a Universal Background Model and saves it to file"""

  # Loads the data into an Arrayset
  ar = load_data(train_files.itervalues())

  # Computes input size
  input_size = ar.shape[0]

  # Normalizes the Arrayset if required
  if not norm_KMeans:
    normalizedAr = ar
  else:
    (normalizedAr,stdAr) = NormalizeStdArrayset(ar)


  # Creates the machines (KMeans and GMM)
  kmeans = bob.machine.KMeansMachine(n_gaussians, input_size)
  gmm = bob.machine.GMMMachine(n_gaussians, input_size)

  # Creates the KMeansTrainer
  kmeansTrainer = bob.trainer.KMeansTrainer()
  kmeansTrainer.convergenceThreshold = convergence_threshold
  kmeansTrainer.maxIterations = iterk

  # Trains using the KMeansTrainer
  kmeansTrainer.train(kmeans, normalizedAr)

  [variances, weights] = kmeans.getVariancesAndWeightsForEachCluster(normalizedAr)
  means = kmeans.means

  # Undoes the normalization
  if norm_KMeans:
    multiplyVectorsByFactors(means, stdAr)
    multiplyVectorsByFactors(variances, stdAr ** 2)

  # Initializes the GMM
  gmm.means = means
  gmm.variances = variances
  gmm.weights = weights
  gmm.setVarianceThresholds(variance_threshold)

  # Trains the GMM
  trainer = bob.trainer.ML_GMMTrainer(update_means, update_variances, update_weights)
  trainer.convergenceThreshold = convergence_threshold
  trainer.maxIterations = iterg
  trainer.train(gmm, ar)

  # Saves the UBM to file
  gmm.save(bob.io.HDF5File(ubm_filename))


def gmm_stats(features_input, ubm_filename, gmmstats_output, force=False):
  """Computes GMM statistics against a UBM"""
  # Loads the UBM 
  if not os.path.exists(ubm_filename):
    raise RuntimeError, "Cannot find UBM %s" % (ubm_filename)
  ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_filename))    

  # Initializes GMMStats object 
  gmmstats = bob.machine.GMMStats(ubm.nGaussians, ubm.nInputs)

  # Processes the 'dictionary of files'
  for k in features_input:
    # Removes old file if required
    if force == True and os.path.exists(gmmstats_output[k]):
      print "Remove old statistics %s." % (gmmstats_output[k])
      os.remove(gmmstats_output[k])

    if os.path.exists(gmmstats_output[k]):
      print "GMM statistics %s already exists."  % (gmmstats_output[k])
    else:
      print "Computing statistics from features %s." % (features_input[k])
      # Loads input features file
      features = bob.io.Arrayset( str(features_input[k]) )
      # Accumulates statistics
      gmmstats.init()
      ubm.accStatistics(features, gmmstats)
      # Saves the statistics
      utils.ensure_dir(os.path.dirname( str(gmmstats_output[k]) ))
      gmmstats.save(bob.io.HDF5File( str(gmmstats_output[k]) ))


def gmm_enrol_model(enrol_files, model_path, ubm_filename,
          iterg=1, convergence_threshold=0.0005, variance_threshold=0.0005, relevance_factor=4, 
          responsibilities_threshold=0, adapt_weight=False, adapt_variance=False, torch3_map=False, alpha_torch3=0.5):
  """Enrols a GMM using MAP adaptation"""
  # Loads the data into an Arrayset
  ar = load_data(enrol_files.itervalues())

  # Loads the UBM/prior gmm
  if not os.path.exists(ubm_filename):
      raise RuntimeError, "Cannot find UBM %s" % (ubm_filename)
  ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_filename))    
  ubm.setVarianceThresholds(variance_threshold)

  # Creates the trainer
  if responsibilities_threshold == 0.:
    trainer = bob.trainer.MAP_GMMTrainer(relevance_factor, True, adapt_variance, adapt_weight)
  else:
    trainer = bob.trainer.MAP_GMMTrainer(relevance_factor, True, adapt_variance, adapt_weight, responsibilities_threshold)
  trainer.convergenceThreshold = convergence_threshold
  trainer.maxIterations = iterg
  trainer.setPriorGMM(ubm)

  if torch3_map:
    trainer.setT3MAP(alpha_torch3)

  # Creates a GMM from the UBM
  gmm = bob.machine.GMMMachine(ubm)
  gmm.setVarianceThresholds(variance_threshold)

  # Trains the GMM
  trainer.train(gmm, ar)

  # Saves it to the given file
  gmm.save(bob.io.HDF5File(model_path))


def gmm_scores_A(models_ids, models_dir, probe_files, ubm_filename, db,
                 zt_norm_A_dir, scores_nonorm_dir, group, probes_split_id):
  """Computes a split of the A matrix for the ZT-Norm and saves the raw scores to file"""
  
  # Loads the UBM 
  if not os.path.exists(ubm_filename):
      raise RuntimeError, "Cannot find UBM %s" % (ubm_filename) 
  ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_filename))    

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
    models = [bob.machine.GMMMachine(bob.io.HDF5File(model_path))]
    clients_ids = [db.getClientIdFromModelId(model_id)]

    # Saves the A row vector for each model and Z-Norm samples split
    A = bob.machine.linearScoring(models, ubm, probe_tests)
    bob.io.save(A, os.path.join(zt_norm_A_dir, group, str(model_id) + "_" + str(probes_split_id).zfill(4) + ".hdf5"))

    # Saves to text file
    import utils
    scores_list = utils.convertScoreToList(np.reshape(A, A.size), probe_files)
    sc_nonorm_filename = os.path.join(scores_nonorm_dir, group, str(model_id) + "_" + str(probes_split_id).zfill(4) + ".txt")
    f_nonorm = open(sc_nonorm_filename, 'w')
    for x in scores_list:
      f_nonorm.write(str(x[2]) + " " + str(x[0]) + " " + str(x[3]) + " " + str(x[4]) + "\n")
    f_nonorm.close()


def gmm_ztnorm_B(models_ids, models_dir, zfiles, ubm_filename, db,
                 zt_norm_B_dir, group, zsamples_split_id):
  """Computes a split of the B matrix for the ZT-Norm"""
  
  # Loads the UBM 
  if not os.path.exists(ubm_filename):
      raise RuntimeError, "Cannot find UBM %s" % (ubm_filename) 
  ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_filename))    

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
    models = [bob.machine.GMMMachine(bob.io.HDF5File(model_path))]

    # Save the B row vector for each model and Z-Norm samples split
    B = bob.machine.linearScoring(models, ubm, znorm_tests)
    bob.io.save(B, os.path.join(zt_norm_B_dir, group, str(model_id) + "_" + str(zsamples_split_id).zfill(4) + ".hdf5"))


def gmm_ztnorm_C(tmodel_id, tnorm_models_dir, probe_files, ubm_filename, db,
                 zt_norm_C_dir, group, probes_split_id):
  """Computes a split of the C matrix for the ZT-Norm"""
  
  # Loads the UBM 
  if not os.path.exists(ubm_filename):
      raise RuntimeError, "Cannot find UBM %s" % (ubm_filename) 
  ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_filename))    

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
  tmodels = [bob.machine.GMMMachine(bob.io.HDF5File(tmodel_path))]

  # Saves the C row vector for each T-Norm model and samples split
  C = bob.machine.linearScoring(tmodels, ubm, probe_tests)
  bob.io.save(C, os.path.join(zt_norm_C_dir, group, "TM" + str(tmodel_id) + "_" + str(probes_split_id).zfill(4) + ".hdf5"))


def gmm_ztnorm_D(tnorm_models_ids, tnorm_models_dir, zfiles, ubm_filename, db,
                 zt_norm_D_dir, zt_norm_D_sameValue_dir, group, zsamples_split_id):
  """Computes a split of the D matrix for the ZT-Norm"""
  
  # Loads the UBM 
  if not os.path.exists(ubm_filename):
      raise RuntimeError, "Cannot find UBM %s" % (ubm_filename) 
  ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_filename))    

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
    tnorm_models = [bob.machine.GMMMachine(bob.io.HDF5File(tmodel_path))]
    tnorm_clients_ids = [db.getClientIdFromModelId(tmodel_id)]

    # Save the D and D_sameValue row vector for each T-Norm model and Z-Norm samples split
    D_tm = bob.machine.linearScoring(tnorm_models, ubm, znorm_tests)
    bob.io.save(D_tm, os.path.join(zt_norm_D_dir, group, str(tmodel_id) + "_" + str(zsamples_split_id).zfill(4) + ".hdf5"))
    D_sameValue_tm = bob.machine.ztnormSameValue(tnorm_clients_ids, znorm_clients_ids)
    bob.io.save(D_sameValue_tm, os.path.join(zt_norm_D_sameValue_dir, group, str(tmodel_id) + "_" + str(zsamples_split_id).zfill(4) + ".hdf5"))
