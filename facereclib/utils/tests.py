#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu Jul 19 17:09:55 CEST 2012
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

from .resources import resource_keys, load_resource

def configuration_file(name, type, dir = None):
  # test if the resource is known
  if name in resource_keys(type):
    # resource registered, just load it
    return load_resource(name, type)
  else: # resource not registered, but available...
    # import resource (actually this is a hack, but better than dealing with file names...)
    exec "from facereclib.configurations." + dir + " import " + name
    # get the database defined in the resource
    return eval(name + "." + type)
