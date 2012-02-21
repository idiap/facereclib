#!/bin/bash

cd /idiap/home/mguenther/Source/tools/facereclib/calls

C=/idiap/home/mguenther/Source/tools/facereclib/config

if [ ! -d dct_gmm ]; then
  mkdir dct_gmm
fi

cd dct_gmm
#  ../../shell.py /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py -- -b dct_gmm -d $C/database/banca_P.py -t $C/tools/dct_gmm.py -p $C/features/dct_blocks.py -s ubm_gmm $*
  ../../shell.py /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py -- -b dct_gmm_grid -d $C/database/banca_P.py -t $C/tools/dct_gmm.py -p $C/features/dct_blocks.py -s ubm_gmm -g $C/grid/demanding.py $*
cd ..


