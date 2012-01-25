#!/bin/bash 
# Andre Anjos <andre.dos.anjos@gmail.com>
# Mon 28 Jun 2010 15:43:17 CEST

fdel () {
  find $1 -name "$2" -print0 | xargs -0 rm -vf
}

fdel . "*~"
fdel . "*.pyc"
fdel logs "*.e*"
fdel logs "*.o*"
