#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Features for speaker recognition"""

import numpy,math
import bob
import os
import time




class Cepstral:
  """Extracts Cepstral coefficents"""
  def __init__(self, 
      requires_training = False, # enable, if your extractor needs training
      win_length_ms = 20,
      win_shift_ms = 10,
      n_filters = 24,
      dct_norm = False,
      f_min = 0.0,
      f_max = 4000,
      delta_win = 2,
      mel_scale = True,
      with_energy = True,
      with_delta = True,
      with_delta_delta = True,
      with_delta_energy = True,
      with_delta_delta_energy = True,
      n_ceps = 19,
      pre_emphasis_coef = 0.95,
      with_vad_filtering = True,
      use_mod_4hz = False,
      existing_mod_4hz_path = '', 
      win_shift_ms_2 = 16, 
      threshold = 1.1, 
      use_existing_vad = False,
      existing_vad_path = '',
      normalize_features = True,
      alpha = 2,
      max_iterations = 10
  ):
    
    self.requires_training = requires_training
    self.m_win_length_ms = win_length_ms
    self.m_win_shift_ms = win_shift_ms
    self.m_n_filters = n_filters
    self.m_dct_norm = dct_norm
    self.m_f_min = f_min
    self.m_f_max = f_max
    self.m_delta_win = delta_win
    self.m_mel_scale = mel_scale
    self.m_with_energy = with_energy
    self.m_with_delta = with_delta
    self.m_with_delta_delta = with_delta_delta
    self.m_with_delta_energy = with_delta_energy
    self.m_with_delta_delta_energy = with_delta_delta_energy
    self.m_nceps = n_ceps
    self.m_pre_emphasis_coef = pre_emphasis_coef
    self.m_with_vad_filtering = with_vad_filtering
    self.m_use_mod_4zz = use_mod_4hz #existingMod4HzPath = '/idiap/user/ekhoury/LOBI/work/Modulation_4Hz/banca/4Hz/'
    self.m_win_shift_ms_2 = win_shift_ms_2 # for the modulation energy
    self.m_threshold = threshold # for the modulation energy
    self.m_use_existing_vad = use_existing_vad #existingVADPath = '/idiap/temp/ekhoury/SRE2012/work/I4U/IDIAP/DATA/vad/exp/' # should be precised if useExistingVAD is equal True
    self.m_normalize_features = normalize_features
    self.m_alpha = alpha
    self.m_max_iterations = max_iterations
    self.m_features_mask = numpy.concatenate((numpy.arange(0,self.m_nceps), numpy.arange(self.m_nceps,60)))
    

  numpy.set_printoptions(precision=2, threshold=numpy.nan, linewidth=200)

  
  def _normalize_std_array(self, vector):
    """Applies a unit variance normalization to an arrayset"""

    # Initializes variables
    length = 1
    n_samples = len(vector)
    mean = numpy.ndarray((length,), 'float64')
    std = numpy.ndarray((length,), 'float64')

    mean.fill(0)
    std.fill(0)

    # Computes mean and variance
    for array in vector:
      x = array.astype('float64')
      mean += x
      std += (x ** 2)

    mean /= n_samples
     
    std /= n_samples
    std -= (mean ** 2)
    std = std ** 0.5 

    arrayset = numpy.ndarray(shape=(n_samples,mean.shape[0]), dtype=numpy.float64);
    
    for i in range (0, n_samples):
      arrayset[i,:] = (vector[i]-mean) / std 
    return arrayset


  def voice_activity_detection(self, params):
    #########################
    ## Initialisation part ##
    #########################
    index = self.m_nceps
    max_iterations = self.m_max_iterations
    alpha = self.m_alpha

    energy_array = numpy.array([row[index] for row in params])
    n_samples = len(energy_array)
    
    normalized_energy = self._normalize_std_array(energy_array)
    
    kmeans = bob.machine.KMeansMachine(2, 1)
    m_ubm = bob.machine.GMMMachine(2, 1)
      
    kmeans_trainer = bob.trainer.KMeansTrainer()
    kmeans_trainer.convergence_threshold = 0.0005
    kmeans_trainer.max_iterations = max_iterations;
    kmeans_trainer.check_no_duplicate = True
  
    # Trains using the KMeansTrainer
    kmeans_trainer.train(kmeans, normalized_energy)
    
    [variances, weights] = kmeans.get_variances_and_weights_for_each_cluster(normalized_energy)
    means = kmeans.means
    print "means = ", means[0], means[1]
    print "variances = ", variances[0], variances[1]
    print "weights = ", weights[0], weights[1]
    
    # Initializes the GMM
    m_ubm.means = means
    
    m_ubm.variances = variances
    m_ubm.weights = weights
    m_ubm.set_variance_thresholds(0.0005)
    
    trainer = bob.trainer.ML_GMMTrainer(True, True, True)
    trainer.convergence_threshold = 0.0005
    trainer.max_iterations = 25
    trainer.train(m_ubm, normalized_energy)
    means = m_ubm.means
    weights = m_ubm.weights
    print "means = ", means[0], means[1]
    print "weights = ", weights[0], weights[1]
    
    if means[0] < means[1]:
      higher = 1;
      lower = 0;
    else:
      higher = 0;
      lower = 1;
    
    
    label = numpy.array(numpy.ones(n_samples), dtype=numpy.int16);
    
    higher_mean_gauss = m_ubm.get_gaussian(higher);
    lower_mean_gauss = m_ubm.get_gaussian(lower);

    k=0;
    for i in range(n_samples):
      if higher_mean_gauss.log_likelihood(normalized_energy[i]) < lower_mean_gauss.log_likelihood( normalized_energy[i]):
        label[i]=0
      else:
        label[i]=label[i] * 1
    print "After Energy-based VAD there are ", numpy.sum(label), " remaining over ", len(label)
    
    out_params = numpy.ndarray(shape=((label == 1).sum(),len(self.m_features_mask)), dtype=numpy.float64)
    i=0;
    cur_i=0;
   
    for row in params:
      if label[i]==1:
        for k in range(len(self.m_features_mask)):
          out_params[cur_i,k] = row[self.m_features_mask[k]]
        cur_i = cur_i + 1
      i = i+1;
    print i
  
    return out_params
    
  ####################################
  ###    End of the Core Code      ###
  ####################################
  
  def normalize_features(self, params):
  #########################
  ## Initialisation part ##
  #########################
  
    normalized_vector = [ [ 0 for i in range(params.shape[1]) ] for j in range(params.shape[0]) ] ;
    for index in range(params.shape[1]):
      vector = numpy.array([row[index] for row in params])
      n_samples = len(vector)
      norm_vector = self._normalize_std_array(vector)
      
      for i in range(n_samples):
        normalized_vector[i][index]=numpy.asscalar(norm_vector[i]);    
    data = numpy.array(normalized_vector)
    return data
  
  
  def modulation_4Hz(self, inFile4Hz, n_samples, win_shift_1, win_shift_2, Threshold):
    #typically, win_shift_1 = 10ms, win_shift_2 =16ms
    f=open(inFile4Hz);
    list_1s_shift=[[float(i) for i in line.split()] for line in open(inFile4Hz)];
   
    len_list=len(list_1s_shift);
    valeur_16ms = numpy.array(numpy.zeros(len_list, dtype=numpy.float));
    
    valeur_16ms[0]=numpy.array(list_1s_shift[0]);
    for j in range(2, 63):
      valeur_16ms[j-1]=((j-1.0)/j)*valeur_16ms[j-2] +(1.0/j)*numpy.array(list_1s_shift[j-1]);
    
        
    for j in range(63, len_list-63):
      valeur_16ms[j-1]=numpy.array(numpy.mean(list_1s_shift[j-62:j]))
    
    
    valeur_16ms[len_list-1] = numpy.mean(list_1s_shift[len_list -1])
    for j in range(2, 63):
      valeur_16ms[len_list-j]=((j-1.0)/j)*valeur_16ms[len_list+1-j] +(1.0/j)*numpy.array(list_1s_shift[len_list-j]);
    
    label = numpy.array(numpy.zeros(n_samples), dtype=numpy.int16);
    
    Mod_4Hz = numpy.array(numpy.zeros(n_samples, dtype=numpy.float));
    for x in range(0, n_samples):
      y = int (win_shift_1 * x / win_shift_2);
      r =  (win_shift_1 * x) % win_shift_2;
      
      Mod_4Hz[x] = (1.0 - r) * valeur_16ms[numpy.minimum(y, len(valeur_16ms)-1)] + r * valeur_16ms[numpy.minimum(y+1, len(valeur_16ms)-1)];
              
      if Mod_4Hz[x] > Threshold:
        label[x]=1;
      else:
        label[x]=0;
    return Mod_4Hz

  def use_existing_vad(self,inArr, vad_file):
    f=open(vad_file)
    nsamples = len(inArr)
    dimensionality=inArr[0].shape[0]
    ns=0
    for line in f:
      line = line.strip()
      st_frame = float(line.split(' ')[2])
      en_frame = float(line.split(' ')[4])
      st_frame = min(int(st_frame * 100), nsamples)
      st_frame = max(st_frame, 0)
      en_frame = min(int(en_frame * 100), nsamples)
      en_frame = max(en_frame, 0)
      ns=ns+en_frame-st_frame

    outArr = numpy.ndarray(shape=(ns,dimensionality), dtype=numpy.float64)
    c=0
    for line in f:
      line = line.strip()
      st_frame = float(line.split(' ')[2])
      en_frame = float(line.split(' ')[4])
      st_frame = min(int(st_frame * 100), nsamples)
      st_frame = max(st_frame, 0)
      en_frame = min(int(en_frame * 100), nsamples)
      en_frame = max(en_frame, 0)
      for i in range(st_frame, en_frame):
        outArr[c,:]=inArr[i]
        c=c+1
    return outArr   


  def __call__(self, rate_wavsample, annotations = None):
    """Computes and returns normalized cepstral features for the given input wave file"""
    
    #print "Input file : ", rate_wavsample
    #rate_wavsample = self._read(input_file)
    
    # Feature extraction
    
    # Set parameters
    wl = self.m_win_length_ms
    ws = self.m_win_shift_ms
    nf = self.m_n_filters
    nc = self.m_nceps

    f_min = self.m_f_min
    f_max = self.m_f_max
    dw = self.m_delta_win
    pre = self.m_pre_emphasis_coef


    print rate_wavsample[0]
    print wl
    print ws
    print nf
    print nc
    print f_min
    print f_max
    print dw
    print pre
    ceps = bob.ap.Ceps(rate_wavsample[0], wl, ws, nf, nc, f_min, f_max, dw, pre)
    

    ceps.dct_norm = self.m_dct_norm
    ceps.mel_scale = self.m_mel_scale
    ceps.with_energy = self.m_with_energy
    ceps.with_delta = self.m_with_delta
    ceps.with_delta_delta = self.m_with_delta_delta
    
    
    cepstral_features = ceps(rate_wavsample[1] )
   
    # Voice activity detection
    if self.m_with_vad_filtering:
      filtered_features = self.voice_activity_detection(cepstral_features)
    else:
      filtered_features = cepstral_features

    if self.m_normalize_features:
      normalized_features = self.normalize_features(filtered_features)
    else:
      normalized_features = filtered_features
    
    return normalized_features


  def read_feature(self, feature_file):
    return bob.io.load(feature_file)


  def save_feature(self, feature, feature_file):
        bob.io.save(feature, feature_file)

  def load(self, extractor_file):
    """Loads the parameters required for feature extraction from the extractor file.
    This function usually is only useful in combination with the 'train' function (see below).
    In this base class implementation, it does nothing.
    """
    pass
