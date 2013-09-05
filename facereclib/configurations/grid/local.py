import facereclib

# define the queue using all the default parameters
grid = facereclib.utils.GridParameters(
  grid = 'local',
  number_of_parallel_processes = 4
)

# define a queue that is highly parallelized
grid_p16 = facereclib.utils.GridParameters(
  number_of_preprocessings_per_job = 50,
  number_of_extracted_features_per_job = 50,
  number_of_projected_features_per_job = 50,
  number_of_enrolled_models_per_job = 10,
  number_of_models_per_scoring_job = 10,

  grid = 'local',
  number_of_parallel_processes = 4
)