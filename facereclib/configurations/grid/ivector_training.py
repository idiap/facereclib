import facereclib

# define a queue with parameters to train ISV
grid = facereclib.utils.GridParameters(
  training_queue = 'default',
  # preprocessing
  number_of_preprocessings_per_job = 1000,
  # feature extraction
  number_of_extracted_features_per_job = 1000,
  extraction_queue = 'default',
  # feature projection
  number_of_projected_features_per_job = 200,
  projection_queue = 'default',
  # model enrollment
  number_of_enrolled_models_per_job = 20,
  enrollment_queue = 'default',
  # scoring
  number_of_models_per_scoring_job = 20,
  scoring_queue = 'default'
)
# add special queue parameters for the I-Vector training
grid.ivec_training_queue = grid.queue('default')

