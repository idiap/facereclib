#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu May 24 10:41:42 CEST 2012
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


import facereclib

cls = None
# Here we test all installed databases and everything else that has a bob.test declared in the setup.py
# The technology is adapted from the bob.db.aggegator tests
import pkg_resources
for i, ep in enumerate(pkg_resources.iter_entry_points('bob.test')):
  ep_parts = str(ep).split(' = ')[1].split(':')
  if 'xfacereclib' in ep_parts[0]:
    facereclib.utils.debug("Collected external test '%s' from '%s'"% (ep_parts[1], ep_parts[0]))
    cls = ep.load()
    exec('Test%d = cls' % i)
    exec('Test%d.__name__ = "%s [%s:%s]"' % (i, ep_parts[1], ep_parts[0], ep_parts[1]))

# clean-up since otherwise the last test is re-run
del cls



