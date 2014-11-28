#!/usr/bin/env python

import facereclib
import bob.ip.gabor

similarity_function = bob.ip.gabor.Similarity("PhaseDiffPlusCanberra", bob.ip.gabor.Transform())

def gabor_jet_similarities(f1, f2):
  """Computes the similarity vector between two Gabor graph features"""
  assert len(f1) == len(f2)
  return [similarity_function(f1[i], f2[i]) for i in range(len(f1))]


tool = facereclib.tools.BIC(
    # measure to compare two features in input space
    comparison_function = gabor_jet_similarities,
    # load and save functions
    load_function = bob.ip.gabor.load_jets,
    save_function = bob.ip.gabor.save_jets,
    # Limit the number of training pairs
    maximum_training_pair_count = 1000000,
    # Dimensions of intrapersonal and extrapersonal subspaces
    subspace_dimensions = (20, 20),
    multiple_model_scoring = 'max'
)
