#!/bin/bash

cd /idiap/home/mguenther/Source/tools/facereclib/calls

C=/idiap/home/mguenther/Source/tools/facereclib/config

if [ ! -d plda ]; then
  mkdir plda
fi

cd plda
#  ../../shell.py /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py -- -b plda_local -d $C/database/banca_P.py -t $C/tools/plda.py -p $C/features/eigenfaces.py -s eigenfaces $*
  ../../shell.py /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py -- -b plda -d $C/database/banca_P.py -t $C/tools/plda.py -p $C/features/eigenfaces.py -s eigenfaces -g $C/grid/grid.py $*
cd ..


