#!/bin/sh

# Author: Mark Harvey


# folders
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}


# list of GPUs to use
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="0"

# convert dataset to TFRecords
# this only needs to be run once
python -u images_to_tfrec.py -bd ${BUILD} 2>&1 | tee ${LOG}/tfrec.log


# training
python -u train.py -bd ${BUILD} 2>&1 | tee ${LOG}/train.log

