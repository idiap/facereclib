import xbob.db.atnt
import facereclib
import os

class TestDatabase (facereclib.databases.DatabaseXBobZT):

  def __init__(self):
    # call base class constructor with useful parameters
    facereclib.databases.DatabaseXBobZT.__init__(
        self,
        database = xbob.db.atnt.Database(),
        name = 'test2',
        original_directory = facereclib.utils.tests.atnt_database_directory(),
        original_extension = ".pgm"
    )

  def uses_probe_file_sets(self):
    return True

  def probe_file_sets(self, model_id = None, group = 'dev'):
    """Returns the list of probe File objects (for the given model id, if given)."""
    files = self.arrange_by_client(self.sort(self.m_database.objects(protocol = None, groups = group, purposes = 'probe')))
    # arrange files by clients
    file_sets = []
    for client_files in files:
      # generate file set for each client
      file_set = facereclib.databases.FileSet(client_files[0].client_id, client_files[0].client_id, client_files[0].path)
      file_set.files = client_files
      file_sets.append(file_set)
    return file_sets


  def t_model_ids(self, group = 'dev'):
    return self.model_ids(group)


  def t_enroll_files(self, model_id, group = 'dev'):
    return self.enroll_files(model_id, group)


  def z_probe_files(self, group = 'dev'):
    return self.probe_files(None, group)

  def z_probe_file_sets(self, group = 'dev'):
    return self.probe_file_sets(None, group)

database = TestDatabase()

