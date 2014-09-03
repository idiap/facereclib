import facereclib

# define a queue specifically for the bob.db.lfw database
grid = facereclib.utils.GridParameters(
  training_queue = '16G',
  # model enrollment
  number_of_enrollment_jobs = 16,
  enrollment_queue = '8G',
  # scoring
  number_of_scoring_jobs = 16,
  scoring_queue = '8G'
)
