import xbob.db.atnt
import facereclib
import os

image_directory = os.environ['ATNT_DATABASE_DIRECTORY'] if 'ATNT_DATABASE_DIRECTORY' in os.environ else "/idiap/group/biometric/databases/orl/"


class TestDatabase (facereclib.databases.DatabaseXBobZT):

  def __init__(self):
    # call base class constructor with useful parameters
    facereclib.databases.DatabaseXBobZT.__init__(
        self,
        database = xbob.db.atnt.Database(),
        name = 'test',
        image_directory = image_directory,
        image_extension = ".pgm"
    )

  def t_model_ids(self, group = 'dev'):
    return self.model_ids(group)


  def t_enroll_files(self, model_id, group = 'dev'):
    return self.enroll_files(model_id, group)


  def z_probe_files(self, group = 'dev'):
    return self.probe_files(None, group)

database = TestDatabase()

