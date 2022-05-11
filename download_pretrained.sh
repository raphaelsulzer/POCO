#!/bin/bash

echo "Downloading pretrained models..."

if gdown 1g6KNv_lhnxkyugA5D6__6LyVA7KPUk9k; then
  unzip convonet.zip
  mv convonet out
  rm convonet.zip
  echo "Done!"
else
  echo "Please install gdown with 'conda install -c conda-forge gdown'"
fi