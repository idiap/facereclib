import facereclib

# define a queue with very demanding parameters
grid = facereclib.utils.GridParameters(
  training_queue = '64G',
  # preprocessing
  number_of_preprocessing_jobs = 100,
  preprocessing_queue = '4G-io-big',
  # feature extraction
  number_of_extraction_jobs = 100,
  extraction_queue = '8G-io-big',
  # feature projection
  number_of_projection_jobs = 100,
  projection_queue = '8G-io-big',
  # model enrollment
  number_of_enrollment_jobs = 100,
  enrollment_queue = '8G-io-big',
  # scoring
  number_of_scoring_jobs = 100,
  scoring_queue = '8G-io-big'
)
