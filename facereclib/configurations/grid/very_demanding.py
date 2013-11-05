import facereclib

# define a queue with very demanding parameters
grid = facereclib.utils.GridParameters(
  training_queue = '64G',
  # preprocessing
  number_of_preprocessings_per_job = 100,
  preprocessing_queue = '4G',
  # feature extraction
  number_of_extracted_features_per_job = 100,
  extraction_queue = '8G-io-big',
  # feature projection
  number_of_projected_features_per_job = 20,
  projection_queue = '8G-io-big',
  # model enrollment
  number_of_enrolled_models_per_job = 2,
  enrollment_queue = '8G-io-big',
  # scoring
  number_of_models_per_scoring_job = 1,
  scoring_queue = '8G-io-big'
)
