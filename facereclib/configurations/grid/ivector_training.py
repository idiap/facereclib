import facereclib

# define a queue with parameters to train ISV
queue = facereclib.utils.GridParameters(
  training_queue = '4G-io-big',
  # preprocessing
  number_of_preprocessings_per_job = 1000,
  # feature extraction
  number_of_extracted_features_per_job = 1000,
  extraction_queue = '2G',
  # feature projection
  number_of_projected_features_per_job = 200,
  projection_queue = '2G',
  # model enrollment
  number_of_enrolled_models_per_job = 20,
  enrollment_queue = '4G-io-big',
  # scoring
  number_of_models_per_scoring_job = 20,
  scoring_queue = '4G-io-big'
)
# add special queue parameters for the I-Vector training
queue.ivec_training_queue = queue.queue('Week')

