import facereclib

# define the queue using all the default parameters
grid = facereclib.utils.GridParameters(
  grid = 'local',
  number_of_parallel_processes = 4
)


# define a queue that is highly parallelized
grid_p8 = facereclib.utils.GridParameters(
  grid = 'local',
  number_of_parallel_processes = 8
)

# define a queue that is highly parallelized
grid_p16 = facereclib.utils.GridParameters(
  grid = 'local',
  number_of_parallel_processes = 16
)
