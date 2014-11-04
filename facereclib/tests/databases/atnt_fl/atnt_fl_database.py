import bob.db.verification.filelist
import facereclib
import os

class TestDatabase (facereclib.databases.DatabaseFileList):

  def __init__(self):

    # call base class constructor with useful parameters
    facereclib.databases.DatabaseFileList.__init__(
        self,
        database = bob.db.verification.filelist.Database(
            base_dir = os.path.realpath(os.path.dirname(__file__)),
            original_directory = facereclib.utils.tests.atnt_database_directory(),
            original_extension = ".pgm",
            dev_subdir = '.',
            eval_subdir = '.',
            world_filename = 'world.lst',
            models_filename = 'models.lst',
            probes_filename = 'probes.lst',
            tnorm_filename = 'models.lst',
            znorm_filename = 'probes.lst',
            keep_read_lists_in_memory = True
        ),
        name = 'test_fl',
        protocol = None
    )

database = TestDatabase()

