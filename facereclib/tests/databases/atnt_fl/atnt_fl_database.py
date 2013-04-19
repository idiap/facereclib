import xbob.db.faceverif_fl
import facereclib
import os

class TestDatabase (facereclib.databases.DatabaseXBobZT):

  def __init__(self):
    # call base class constructor with useful parameters
    facereclib.databases.DatabaseXBobZT.__init__(
        self,
        database = xbob.db.faceverif_fl.Database(
            base_dir = os.path.realpath(os.path.dirname(__file__)),
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
        image_directory = "/idiap/group/biometric/databases/orl/",
        image_extension = ".pgm"
    )

database = TestDatabase()

