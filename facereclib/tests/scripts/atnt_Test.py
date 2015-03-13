import bob.db.atnt
import facereclib
import os

class TestDatabase (facereclib.databases.DatabaseBobZT):

  def __init__(self):
    # call base class constructor with useful parameters
    facereclib.databases.DatabaseBobZT.__init__(
        self,
        database = bob.db.atnt.Database(
            original_directory = facereclib.utils.tests.atnt_database_directory()
        ),
        name = 'test',
        check_original_files_for_existence = True
    )


  def all_files(self, groups = ['dev']):
    return facereclib.databases.DatabaseBob.all_files(self, groups)


  def t_model_ids(self, group = 'dev'):
    return self.model_ids(group)


  def t_enroll_files(self, model_id, group = 'dev'):
    return self.enroll_files(model_id, group)


  def z_probe_files(self, group = 'dev'):
    return self.probe_files(None, group)

database = TestDatabase()
