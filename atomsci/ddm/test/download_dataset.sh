#!/bin/bash

cd unit
bash download_dataset.sh
cd ..

cd integrative/delaney_NN
bash download_dataset.sh
cd ../..

cd integrative/delaney_RF
bash download_dataset.sh
cd ../..

cd integrative/wenzel_NN
bash download_dataset.sh
cd ../..
