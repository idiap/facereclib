#!/bin/bash

cd /idiap/home/mguenther/Source/tools/facereclib/calls

C=/idiap/home/mguenther/Source/tools/facereclib/config

if [ ! -d pca ]; then
  mkdir pca
fi

cd pca
#  ../../shell.py /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py -- -b pca_local -d $C/database/banca_P.py -t $C/tools/pca.py -p $C/features/tan_triggs.py -s eigenfaces --preload-probes $*
  ../../shell.py /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py -- -b pca -d $C/database/banca_P.py -t $C/tools/pca.py -p $C/features/tan_triggs.py -s eigenfaces -g $C/grid/grid.py --preload-probes $*
cd ..



