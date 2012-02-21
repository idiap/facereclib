#!/bin/bash

cd /idiap/home/mguenther/Source/tools/facereclib/calls

C=/idiap/home/mguenther/Source/tools/facereclib/config

if [ ! -d jfa ]; then
  mkdir jfa
fi

cd jfa
  ../../shell.py /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py -- -b dct_jfa -d $C/database/banca_P.py -t $C/tools/jfa.py -p $C/features/dct_blocks.py -g $C/grid/demanding.py $*
cd ..

#  ../shell.py /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py -- -b jfa_local -d $C/database/banca_P.py -t $C/tools/jfa.py -p $C/features/dct_blocks.py $*

