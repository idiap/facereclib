import facereclib

# define a queue with demanding parameters
grid = facereclib.utils.GridParameters(
  training_queue = '32G',
  # preprocessing
  number_of_preprocessings_per_job = 200,
  preprocessing_queue = '4G',
  # feature extraction
  number_of_extracted_features_per_job = 200,
  extraction_queue = '8G',
  # feature projection
  number_of_projected_features_per_job = 200,
  projection_queue = '8G',
  # model enrollment
  number_of_enrolled_models_per_job = 10,
  enrollment_queue = '8G',
  # scoring
  number_of_models_per_scoring_job = 10,
  scoring_queue = '8G'
)
