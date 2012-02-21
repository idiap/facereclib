#!/bin/bash

cd /idiap/home/mguenther/Source/tools/facereclib/calls

C=/idiap/home/mguenther/Source/tools/facereclib/config

if [ ! -d lgbphs ]; then
  mkdir lgbphs
fi

cd lgbphs
#  ../../shell.py /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py -- -b lgbphs_local -d $C/database/banca.py -t $C/tools/lgbphs.py -p $C/features/lgbphs.py -s histo $*
  ../../shell.py /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py -- -b lgbphs -d $C/database/banca.py -t $C/tools/lgbphs.py -p $C/features/lgbphs.py -s histo -g $C/grid/demanding.py $*
cd ..

