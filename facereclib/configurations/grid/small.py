import facereclib

# define a queue for small databases
grid = facereclib.utils.GridParameters(
  training_queue = '8G',
  # preprocessing
  number_of_preprocessings_per_job = 20,
  # feature extraction
  number_of_extracted_features_per_job = 20,
  # feature projection
  number_of_projected_features_per_job = 20,
  # model enrollment
  number_of_enrolled_models_per_job = 5,
  # scoring
  number_of_models_per_scoring_job = 5,
)
