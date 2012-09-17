#!/usr/bin/env python

import facereclib

preprocessor = facereclib.preprocessing.Cepstral

# Cepstral parameters
win_length_ms = 20;
win_shift_ms = 10;
n_filters = 24; 
dct_norm = 1.0;
#dct_norm = math.sqrt(2.0 / n_filters);
f_min = 0.0;
f_max = 4000;
delta_win = 2;
fb_linear = True;
withEnergy = True;
withDelta = True;
withDeltaDelta = True;
withDeltaEnergy = True;
withDeltaDeltaEnergy = True;
n_ceps = 19; # 0-->18
energy_mask = n_ceps # 19;
useMod4Hz = False
# Normalization
import numpy
features_mask = numpy.concatenate((numpy.arange(0,n_ceps), numpy.arange(n_ceps+1,51)))
#mask1 = numpy.concatenate((numpy.arange(0,n_ceps), numpy.arange(n_ceps+1,2*(n_ceps+1)))) # [0-->18, 20-->39]
#features_mask = numpy.concatenate((mask1, numpy.arange(2*(n_ceps+1),3*(n_ceps+1)-1))) #[40-->59]
#mask1 = numpy.concatenate((numpy.arange(0,16), numpy.arange(n_ceps+1,37))) # [0-->18, 20-->39]
#features_mask = numpy.concatenate((mask1, numpy.arange(2*(n_ceps+1),56))) #[40-->59]

# VAD parameters
alpha = 2;
max_iterations = 10;
