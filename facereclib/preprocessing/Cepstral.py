#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Features for speaker recognition"""

import numpy,math
import bob
import os

from .Preprocessor import Preprocessor

class Cepstral (Preprocessor):
  """Extracts Cepstral coefficients"""
  def __init__(self, config):
    Preprocessor.__init__(self)
    raise NotImplementedError("This class needs to be restructured to be able to use it with the current version of the FaceRecLib.")
    self.m_config = config

  def _read(self, filename):
    """Read video.FrameContainer containing preprocessed frames"""
    import scipy.io.wavfile
    rate, data = scipy.io.wavfile.read(str(filename)); # the data is read in its native format
    if data.dtype =='int16':
      data = numpy.cast['float'](data)
    #data = data / 32768;
    return [rate,data]


  numpy.set_printoptions(precision=2, threshold=numpy.nan, linewidth=200)

  # convert to Mel scale
  def _mel(self, value):
      return 2595.0 * math.log10(1 + value / 700.0)

  # convert to mel inverse
  def _mel_inv(self, value):
      return 700.0 * (10 ** (value / 2595.0) - 1)


  def _pre_emphasis(self, frame, win_length, a):
    if (a < 0.0) or (a >= 1.0):
      print "Error: The emphasis coeff. should be between 0 and 1"
    if (a == 0.0):
      return frame;
    else:
      for i in range(win_length - 1, 0, -1):
        frame[i] = frame[i] - a * frame[i - 1];
      frame[0] = (1. - a) * frame[0];
    return frame;

  # hamming windowing
  def _hamming_window(self, vector, hamming_kernel, win_length):
    for i in range(win_length):
      vector[i] = vector[i] * hamming_kernel[i]
    return vector

  # compute energy and, if flag is set to true, normalise the frame values by dividing to the energy
  def _sig_norm(self, win_length, frame, flag):
    gain = 0.0
    for i in range(win_length):
      gain = gain + frame[i] * frame[i]
    #gain = math.sqrt(gain)
    ENERGY_FLOOR = 1.0;
    if gain < ENERGY_FLOOR:
      gain = math.log(ENERGY_FLOOR)
    else:
      gain = math.log(gain)

    if(flag and gain != 0.0):
      for i in range(win_length):
        frame[i] = frame[i] / gain
    return gain


  def _log_filter_bank(self, x, n_filters, p_index, win_size):
    x1 = numpy.array(x, dtype=numpy.complex128);

    complex_ = bob.sp.fft(x1);

    for i in range(0, win_size / 2):
      re = complex_[i].real;
      im = complex_[i].imag;
      x[i] = math.sqrt(re * re + im * im);

    #print  x

    filters = self._log_triangular_bank(x, n_filters, p_index);
    #print filters
    return filters;


  def _log_triangular_bank(self, data, n_filters, p_index):
    filters = numpy.zeros(n_filters)
    for i in range(0, n_filters):
      res = 0.0;

      a = 1.0 / (p_index[i + 1] - p_index[i] + 1);

      for j in range(p_index[i], p_index[i + 1]):
        res = res + data[j] * (1.0 - a * (p_index[i + 1] - j));

      a = 1.0 / (p_index[i + 2] - p_index[i + 1] + 1);

      for j in range(p_index[i + 1], p_index[i + 2] + 1):
        res = res + data[j] * (1.0 - a * (j - p_index[i + 1]))

      FBANK_OUT_FLOOR = 1.0;
      if (res < FBANK_OUT_FLOOR):
        filters[i] = math.log(FBANK_OUT_FLOOR);
      else:

        filters[i] = math.log(res)

    return filters

  def _dct_transform(self, filters, n_filters, dct_kernel, n_ceps, dct_norm):
    ceps = numpy.zeros(n_ceps + 1);
    for i in range(1, n_ceps + 1):
      ceps[i - 1] = 0.0;
      for j in range(1, n_filters + 1):
        ceps[i - 1] = ceps[i - 1] + filters[j - 1] * dct_kernel[i - 1][j - 1]
      ceps[i - 1] = ceps[i - 1] * dct_norm;
    #print dct_norm
    #print ceps
    return ceps


  def cepstral_features_extraction(self, rate_wavsample):
    #########################
    ## Initialisation part ##
    #########################


    win_length_ms = self.m_config.win_length_ms
    win_shift_ms = self.m_config.win_shift_ms
    n_filters = self.m_config.n_filters
    n_ceps = self.m_config.n_ceps
    dct_norm = self.m_config.dct_norm
    f_min = self.m_config.f_min
    f_max = self.m_config.f_max
    delta_win = self.m_config.delta_win
    fb_linear = self.m_config.fb_linear
    withEnergy = self.m_config.withEnergy
    withDelta = self.m_config.withDelta
    withDeltaDelta = self.m_config.withDeltaDelta
    withDeltaEnergy = self.m_config.withDeltaEnergy
    withDeltaDeltaEnergy = self.m_config.withDeltaDeltaEnergy

    sf = rate_wavsample[0]
    data = rate_wavsample[1]
    #fid = open(outfilename, 'w')
    #print infilename

    win_length = int (sf * win_length_ms / 1000);
    #print "win_length = ", win_length
    win_shift = int (sf * win_shift_ms / 1000);
    #print "win_shift = ", win_shift
    win_size = int (2.0 ** math.ceil(math.log(win_length) / math.log(2)));
    m = int (math.log(win_size) / math.log(2));
    #print "win_size = ", win_size

    # Hamming initialisation
    cst = 2 * math.pi / (win_length - 1.0);
    hamming_kernel = numpy.zeros(win_length);

    for i in range(win_length):
      hamming_kernel[i] = (0.54 - 0.46 * math.cos(i * cst))
    # Compute cut-off frequencies

    p_index = numpy.array(numpy.zeros(n_filters + 2), dtype=numpy.int16);
    if(fb_linear):
      #linear scale
      for i in range(n_filters + 2):
        alpha = (i) / (n_filters + 1.0);
        f = f_min * (1.0 - alpha) + f_max * alpha;
        #print f
        p_index[i] = int (round((win_size / (sf * 1.0) * f)));
      #print p_index
    else:
      # Mel scale
      m_max = self._mel(f_max);
      m_min = self._mel(f_min);

      #print f_max, m_max, f_min, m_min
      for i in range(n_filters + 2):
        alpha = ((i) / (n_filters + 1.0));
        f = self._mel_inv(m_min * (1 - alpha) + m_max * alpha);
        #print i, n_filters, alpha, m_min, m_max, f
        factor = f / (sf * 1.0);
        p_index[i] = int (round((win_size) * factor));



    #Cosine transform initialisation

    dct_kernel = [ [ 0 for i in range(n_filters) ] for j in range(n_ceps) ] ;

    for i in range(1, n_ceps + 1):
      for j in range(1, n_filters + 1):
        dct_kernel[i - 1][j - 1] = math.cos(math.pi * i * (j - 0.5) / n_filters);
    #print dct_kernel
    ######################################
    ### End of the Initialisation part ###
    ######################################

    ######################################
    ###          Core code             ###
    ######################################


    data_size = data.shape[0];

    #print data_size
    n_frames = 1 + (data_size - win_length) / win_shift;
    #print n_frames

    # create features set
    ceps_sequence = numpy.zeros(n_ceps);
    dim = n_ceps;

    if (withEnergy):
      dim = n_ceps + 1;

    if (withDelta):
      dim = dim + n_ceps;

    if (withDeltaEnergy):
      dim = dim + 1;

    if (withDeltaDelta):
      dim = dim + n_ceps;

    if(withDeltaDeltaEnergy):
      dim = dim + 1;

    #print dim
    params = [ [ 0 for i in range(dim) ] for j in range(n_frames) ] ;

    # compute cepstral coefficients
    for i in range(n_frames):
      # create a frame
      frame = numpy.zeros(win_size);
      som = 0.0;
      for j in range(win_length):
        frame[j] = data[j + i * win_shift]
        som = som + frame[j];

      som = som / win_size

      for j in range(win_size):
        frame[j] = frame[j] - som;

      if (withEnergy):
        energy = self._sig_norm(win_length, frame, False);
      #print energy

      # pre-emphasis filtering
      frame = self._pre_emphasis(frame, win_length, 0.95);

      # Hamming windowing
      frame = self._hamming_window(frame, hamming_kernel, win_length)
      #print frame

      filters = self._log_filter_bank(frame, n_filters, p_index, win_size)
      #print filters

      ceps = self._dct_transform(filters, n_filters, dct_kernel, n_ceps, dct_norm)
      #print ceps

      if (withEnergy):
        d1 = n_ceps + 1;
        #print energy
        ceps[n_ceps] = energy;
          #print ceps
      else:
        d1 = n_ceps;


      # stock the results in params matrix
      for k in range(d1):
        params[i][k] = ceps[k];

    # End of cepstral coefficients computation

    # compute Delta coefficient
    if (withDelta):
      som = 0.0;
      for i in range(1,delta_win+1):
        som = som + i*i;
      som = som *2;
      #print som

      for i in range(n_frames):
        for k in range(n_ceps):
          params[i][d1+k] = 0.0;
          for l in range(1, delta_win+1):
            if (i+l < n_frames):
              p_ind = i+l;
            else:
              p_ind = n_frames - 1;
            if (i-l > 0):
              n_ind = i-l;
            else:
              n_ind = 0;
            params[i][d1+k] = params[i][d1+k] + l * (params[p_ind][k] - params[n_ind][k]);
          params[i][d1+k] = params[i][d1+k] / som;

    # compute Delta of the Energy
    if (withDeltaEnergy):
      som = 0.0;
      for i in range(1,delta_win+1):
        som = som + i*i;
      som = som *2;

      for i in range(n_frames):
        k = n_ceps;
        params[i][d1+k] = 0.0;
        for l in range(1, delta_win+1):
          if (i+l < n_frames):
            p_ind = i+l;
          else:
            p_ind = n_frames - 1;
          if (i-l > 0):
            n_ind = i-l;
          else:
            n_ind = 0;
          params[i][d1+k] = params[i][d1+k] + l* (params[p_ind][k] - params[n_ind][k]);
        params[i][d1+k] = params[i][d1+k] / som;

    # compute Delta Delta of the coefficients
    if (withDeltaDelta):
      som = 0.0;
      for i in range(1,delta_win+1):
        som = som + i*i;
      som = som *2;
      for i in range(n_frames):
        for k in range(n_ceps):
          params[i][2*d1+k] = 0.0;
          for l in range(1, delta_win+1):
            if (i+l < n_frames):
              p_ind = i+l;
            else:
              p_ind = n_frames - 1;
            if (i-l > 0):
              n_ind = i-l;
            else:
              n_ind = 0;
            params[i][2*d1+k] = params[i][2*d1+k] + l * (params[p_ind][d1+k] - params[n_ind][d1+k]);
          params[i][2*d1+k] = params[i][2*d1+k] / som;

    # compute Delta Delta of the energy
    if (withDeltaDeltaEnergy):
      som = 0.0;
      for i in range(1,delta_win+1):
        som = som + i*i;
      som = som *2;

      for i in range(n_frames):
        k = n_ceps
        params[i][2*d1+k] = 0.0;
        for l in range(1, delta_win+1):
          if (i+l < n_frames):
            p_ind = i+l;
          else:
            p_ind = n_frames - 1;
          if (i-l > 0):
            n_ind = i-l;
          else:
            n_ind = 0;
          params[i][2*d1+k] = params[i][2*d1+k] + l * (params[p_ind][d1+k] - params[n_ind][d1+k]);
        params[i][2*d1+k] = params[i][2*d1+k] / som;

    #print params
    #print n_frames, dim
    data = bob.io.Arrayset(params)
    #print data
    return data



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
    #print "mean = ", mean

    std /= n_samples
    std -= (mean ** 2)
    std = std ** 0.5 # sqrt(std)

    #print "std = ", std

    arrayset = bob.io.Arrayset();

    for i in range (0, n_samples):
      arrayset.append( (vector[i]-mean) / std, 'float64')

    return arrayset


  def voice_activity_detection(self, params, inputFile4Hz):
    #########################
    ## Initialisation part ##
    #########################


    index = self.m_config.energy_mask
    max_iterations = self.m_config.max_iterations
    alpha = self.m_config.alpha
    features_mask = self.m_config.features_mask
    useMod4Hz = self.m_config.useMod4Hz
    print "alpha=", alpha
    print "useMod4Hz", useMod4Hz

    energy_array = numpy.array([row[index] for row in params])
    #print energy_array
    n_samples = len(energy_array)

    normalized_energy = self._normalize_std_array(energy_array)
    #print normalized_energy


    kmeans = bob.machine.KMeansMachine(2, 1)
    m_ubm = bob.machine.GMMMachine(2, 1)
      # Creates the KMeansTrainer
    kmeans_trainer = bob.trainer.KMeansTrainer()
    kmeans_trainer.convergence_threshold = 0.0005
    kmeans_trainer.max_iterations = max_iterations;

      # Trains using the KMeansTrainer
    #print " -> Training kmeans"

    kmeans_trainer.train(kmeans, normalized_energy)

    [variances, weights] = kmeans.get_variances_and_weights_for_each_cluster(normalized_energy)
    means = kmeans.means
    print "means = ", means[0], means[1]
    print "variances = ", variances[0], variances[1]
    print "weights = ", weights[0], weights[1]


    # Initializes the GMM
    m_ubm.means = means
    #print "--before : means = ", means
    m_ubm.variances = variances
    m_ubm.weights = weights
    m_ubm.set_variance_thresholds(0.0005)

    trainer = bob.trainer.ML_GMMTrainer(True, True, True)
    trainer.convergence_threshold = 0.0005
    trainer.max_iterations = 25
    trainer.train(m_ubm, normalized_energy)

    means = m_ubm.means
    #print "--after : means = ", means
    #print means
    #print variances
    #print weights

    if means[0] < means[1]:
      higher = 1;
      lower = 0;
    else:
      higher = 0;
      lower = 1;

    Threshold = means[higher] - alpha * numpy.sqrt(variances[higher])
    #print "Threshold = ", Threshold


    if useMod4Hz:
      label = self.modulation_4Hz(inputFile4Hz, n_samples, self.m_config.win_shift_ms, self.m_config.win_shift_ms_2, self.m_config.Threshold)
      print "After Energy Modulation-based VAD there are ", numpy.sum(label), " remaining over ", len(label)
    else:
      label = numpy.array(numpy.ones(n_samples), dtype=numpy.int16);

    higher_mean_gauss = m_ubm.get_gaussian(higher);
    lower_mean_gauss = m_ubm.get_gaussian(lower);

    #print higher_mean_gauss.mean, lower_mean_gauss.mean
    k=0;
    #print params.shape
    for i in range(n_samples):
      if normalized_energy[i]< Threshold:

      #if higher_mean_gauss.log_likelihood(normalized_energy[i]) < lower_mean_gauss.log_likelihood( normalized_energy[i]):
        label[i]=0
      else:
        label[i]=label[i] * 1
      #print normalized_energy[i], higher_mean_gauss.log_likelihood(normalized_energy[i]), lower_mean_gauss.log_likelihood( normalized_energy[i]), label[i], (normalized_energy[i]> Threshold)
    #print label
    print "After Energy-based VAD there are ", numpy.sum(label), " remaining over ", len(label)
    out_params = bob.io.Arrayset();
    i=0;

    for row in params:
      if label[i]==1:
        row2=numpy.array(numpy.zeros(len(features_mask)))
        for k in range(len(row2)):
          row2[k]=row[features_mask[k]]
        #print row
        #print row2

        out_params.append(row2);
        #print out_params
        #time.sleep(10)
      i = i+1;

    return out_params

  ####################################
  ###    End of the Core Code      ###
  ####################################

  def normalize_features(self, params):
  #########################
  ## Initialisation part ##
  #########################

    normalized_vector = [ [ 0 for i in range(params.shape[0]) ] for j in range(numpy.size(params)/params.shape[0]) ] ;

    for index in range(params.shape[0]):
      vector = numpy.array([row[index] for row in params])
      n_samples = len(vector)
      norm_vector = self._normalize_std_array(vector)

      for i in range(n_samples):
        normalized_vector[i][index]=numpy.asscalar(norm_vector[i]);
    data = bob.io.Arrayset(normalized_vector)
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
    #valeur_16ms[2:63]=numpy.arange(1,62)/numpy.arange(2,63) * valeur_16ms[0:61] + 1/numpy.arange(2,63)*list_1s_shift[1:62]

    for j in range(63, len_list-63):
      #mean_val = 0;
      #for  k in range(j-62, j):
      #  mean_val = mean_val + list_1s_shift[k-1];
      #valeur_16ms[j-1]=mean_val/62;

      valeur_16ms[j-1]=numpy.array(numpy.mean(list_1s_shift[j-62:j]))


    valeur_16ms[len_list-1] = numpy.mean(list_1s_shift[len_list -1])
    for j in range(2, 63):
      valeur_16ms[len_list-j]=((j-1.0)/j)*valeur_16ms[len_list+1-j] +(1.0/j)*numpy.array(list_1s_shift[len_list-j]);

    #print valeur_16ms

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

    #print Mod_4Hz
    #print label
    print numpy.sum(label)
    return label



  def read_original_image(self, input_file):
    """Reads the original image (in this case: a wave file) from the given input file"""
    return self._read(input_file)

  def __call__(self, rate_wavsample, annotations = None):
    """Computes and returns normalized cepstral features for the given input wave file"""

    # Computes Cepstral features

    cepstral_features = self.cepstral_features_extraction(rate_wavsample)
    print "Input file : ", input_file
    print "original_features :", numpy.size(cepstral_features)/cepstral_features.shape[0], cepstral_features.shape[0]

    base_filename = os.path.splitext(os.path.basename(input_file))[0];

    input_file_4hz = '/idiap/temp/ekhoury/mobio/4Hz/'+ base_filename + '.4hz'
    print input_file_4hz
    filtered_features = self.voice_activity_detection(cepstral_features, input_file_4hz)
    print "filtered_features :", numpy.size(filtered_features)/filtered_features.shape[0], filtered_features.shape[0]

    normalized_features = self.normalize_features(filtered_features)
    print "normalized_features :", numpy.size(normalized_features)/normalized_features.shape[0], normalized_features.shape[0]
    return normalized_features


  def read_image(self, image_file):
    """Reads the image (in this case, the Arrayset) from file"""
    return bob.io.Arrayset(image_file)
