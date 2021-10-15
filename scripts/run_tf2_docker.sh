#!/bin/sh

# Start the TF2 docker container

# Author: Mark Harvey


HERE=$(pwd -P) # Absolute path of current directory
user=`whoami`
uid=`id -u`
gid=`id -g`

docker run --gpus all --privileged=true -it --rm \
           -u $(id -u):$(id -g) \
           -e USER=$user -e UID=$uid -e GID=$gid \
           -w /workspace \
           -v $HERE:/workspace \
           --network=host \
           tensorflow/tensorflow:2.6.0-gpu \
           bash
