import facereclib

# define a queue specifically for the xbob.db.lfw database
grid = facereclib.utils.GridParameters(
  training_queue = '16G',
  # preprocessing
  number_of_preprocessings_per_job = 1000,
  # feature extraction
  number_of_extracted_features_per_job = 1000,
  # feature projection
  number_of_projected_features_per_job = 1000,
  # model enrollment
  number_of_enrolled_models_per_job = 1000,
  enrollment_queue = '8G',
  # scoring
  number_of_models_per_scoring_job = 1000,
  scoring_queue = '8G'
)
