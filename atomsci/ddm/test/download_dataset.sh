#!/bin/bash

cd unit
./download_dataset.sh
cd ..

cd integrative/delaney_NN
./download_dataset.sh
cd ../..

cd integrative/delaney_RF
./download_dataset.sh
cd ../..

cd integrative/wenzel_NN
./download_dataset.sh
cd ../..
