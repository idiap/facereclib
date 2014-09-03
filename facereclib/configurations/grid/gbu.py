import facereclib

# define a queue specifically for the bob.db.gbu database
grid = facereclib.utils.GridParameters(
  training_queue = '32G',
  # preprocessing
  number_of_preprocessing_jobs = 16,
  preprocessing_queue = '8G',
  # feature extraction
  number_of_extraction_jobs = 16,
  extraction_queue = '8G',
  # feature projection
  number_of_projection_jobs = 16,
  projection_queue = '8G',
  # model enrollment
  number_of_enrollment_jobs = 16,
  enrollment_queue = '8G',
  # scoring
  number_of_scoring_jobs = 16,
  scoring_queue = '8G'
)

