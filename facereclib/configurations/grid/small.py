import facereclib

# define a queue for small databases
grid = facereclib.utils.GridParameters(
  training_queue = '8G',
  # preprocessing
  number_of_preprocessing_jobs = 10,
  # feature extraction
  number_of_extraction_jobs = 10,
  # feature projection
  number_of_projection_jobs = 10,
  # model enrollment
  number_of_enrollment_jobs = 5,
  # scoring
  number_of_scoring_jobs = 10,
)
