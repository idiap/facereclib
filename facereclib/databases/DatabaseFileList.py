#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Wed Oct  3 10:31:51 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from .DatabaseBob import DatabaseBobZT

class DatabaseFileList (DatabaseBobZT):
  """This class should be used whenever you have an :py:class:`bob.db.verification.filelist.Database``."""

  def __init__(
      self,
      database,  # The bob database that is used
      **kwargs  # The default parameters of the base class
  ):
    """
    Parameters of the constructor of this database:

    database : :py:class:`bob.db.verification.filelist.Database`
      The database that provides the actual interface

    kwargs
      Keyword arguments directly passed to the :py:class:`DatabaseBobZT` base class constructor
    """

    DatabaseBobZT.__init__(
        self,
        database = database,
        **kwargs
    )


  def all_files(self, groups = ['dev']):
    """Returns all File objects of the database for the current protocol. If the current protocol is 'None' (a string), None (NoneType) will be used instead"""
    files = self.m_database.objects(protocol = self.protocol if self.protocol != 'None' else None, groups = groups, **self.all_files_options)

    # add all files that belong to the ZT-norm
    for group in groups:
      if group == 'world': continue
      if self.m_database.implements_zt(protocol = self.protocol if self.protocol != 'None' else None, groups = group):
        files += self.m_database.tobjects(protocol = self.protocol if self.protocol != 'None' else None, groups = group, model_ids = None)
        files += self.m_database.zobjects(protocol = self.protocol if self.protocol != 'None' else None, groups = group, **self.m_z_probe_options)
    return self.sort(files)


  def uses_probe_file_sets(self):
    """Defines if, for the current protocol, the database uses several probe files to generate a score."""
    return False


  def model_ids(self, group = 'dev'):
    """Returns the model ids for the given group and the current protocol."""
    return sorted(self.m_database.model_ids(protocol = self.protocol if self.protocol != 'None' else None, groups = group))


  def client_id_from_model_id(self, model_id, group = 'dev'):
    """Returns the client id for the given model id."""
    return self.m_database.get_client_id_from_model_id(model_id, groups = group, protocol = self.protocol if self.protocol != 'None' else None)


  def client_id_from_t_model_id(self, t_model_id, group = 'dev'):
    """Returns the client id for the given T-model id."""
    return self.m_database.get_client_id_from_tmodel_id(t_model_id, groups=group, protocol=self.protocol if self.protocol != 'None' else None)


  def t_model_ids(self, group = 'dev'):
    """Returns the T-Norm model ids for the given group and the current protocol."""
    return sorted(self.m_database.tmodel_ids(protocol = self.protocol if self.protocol != 'None' else None, groups = group))


