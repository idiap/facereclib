#!/bin/bash


FACE_VERIFY="../../shell.py /idiap/home/mguenther/Source/tools/facereclib/script/faceverify.py --"
DB="/idiap/home/$USER/Source/tools/facereclib/config/database/banca_P.py"
FEAT_DIR="/idiap/home/$USER/Source/tools/facereclib/config/features"
TOOL_DIR="/idiap/home/$USER/Source/tools/facereclib/config/tools"
GRID_DIR="/idiap/home/$USER/Source/tools/facereclib/config/grid"

# eigenfaces
if [ ! -d pca ]; then
  mkdir pca
  cd pca
     $FACE_VERIFY -d $DB -t $TOOL_DIR/pca.py -p $FEAT_DIR/tan_triggs.py -g $GRID_DIR/grid.py -b pca -s eigenfaces --preload-probes --preprocessed-image-directory ../preprocessed/TanTriggs $*
  cd ..
fi


# gabor graphs
if [ ! -d gabor ]; then
  mkdir gabor

  cd gabor
   $FACE_VERIFY -d $DB -t $TOOL_DIR/gabor_jet.py -p $FEAT_DIR/grid_graph.py -g $GRID_DIR/grid.py -b gabor -s canberra --preload-probes --preprocessed-image-directory ../preprocessed/FaceEyesNorm $*
   $FACE_VERIFY -d $DB -t $TOOL_DIR/gabor_jet.py -p $FEAT_DIR/grid_graph.py -g $GRID_DIR/grid.py -b gabor -s canberra --preload-probes --preprocessed-image-directory ../preprocessed/FaceEyesNorm $*
  cd ..
fi

# plda on top of eigenfaces
if [ ! -d plda ]; then
  mkdir plda
  cd plda
   $FACE_VERIFY -d $DB -t $TOOL_DIR/plda.py -p $FEAT_DIR/eigenfaces.py -g $GRID_DIR/grid.py -b plda --preload-probes --preprocessed-image-directory ../preprocessed/TanTriggs --skip-preprocessing $*
  cd ..
fi


# UBM/GMM tests on DCT features
if [ ! -d gmm ]; then
  mkdir gmm

  cd gmm
    $FACE_VERIFY -d $DB -t $TOOL_DIR/ubm_gmm.py -p $FEAT_DIR/dct_blocks.py -g $GRID_DIR/demanding.py -b dct_gmm --preprocessed-image-directory ../preprocessed/TanTriggs --skip-preprocessing $*
  cd ..
fi


# LGBPHS
if [ ! -d lgbphs ]; then
  mkdir lgbphs
  cd lgbphs
    $FACE_VERIFY -d $DB -t $TOOL_DIR/lgbphs.py -p $FEAT_DIR/lgbphs.py -g $GRID_DIR/demanding.py -b lgbphs -s intersection --preprocessed-image-directory ../preprocessed/TanTriggs --skip-preprocessing $*
  cd ..
fi

