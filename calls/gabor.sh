#!/bin/bash

cd /idiap/home/mguenther/Source/tools/facereclib/calls

C=/idiap/home/mguenther/Source/tools/facereclib/config
PYTHON=/idiap/group/torch5spro/nightlies/externals/v2/linux-x86_64/bin/python2.6

if [ ! -d gabor ]; then
  mkdir gabor
fi
cd gabor
#  python /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py -b gabor_local -d $C/database/banca_P.py -p $C/features/grid_graph.py -t $C/tools/gabor_jet.py -s canberra_scores $*
  $PYTHON /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py -b gabor -d $C/database/banca_P.py -p $C/features/grid_graph.py -t $C/tools/gabor_jet.py -g $C/grid/grid.py -s canberra_scores $*
cd ..

if [ ! -d gabor_gmm ]; then
  mkdir gabor_gmm
fi  
cd gabor_gmm
#  ../../shell.py /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py -- -b gabor_gmm_new -d $C/database/banca_P.py -p $C/features/grid_graph_dense.py -t $C/tools/ubm_gmm.py -g $C/grid/demanding.py  $*
#  ../../shell.py /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py -- -b gabor_gmm_nonorm -d $C/database/banca_P.py -p $C/features/grid_graph_dense_nonorm.py -t $C/tools/ubm_gmm.py -g $C/grid/demanding.py  $*
cd ..

