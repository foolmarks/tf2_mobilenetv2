#!/bin/sh

# Author: Mark Harvey


# activate the python virtual environment
conda activate vitis-ai-tensorflow2


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


# quantize & evaluate
python -u quantize.py -bd ${BUILD} --evaluate 2>&1 | tee ${LOG}/quantize.log


# compile for selected target board
source compile.sh zcu102 ${BUILD} ${LOG}
source compile.sh u280 ${BUILD} ${LOG}
source compile.sh vck190 ${BUILD} ${LOG}


# make target folders
python -u target.py -t zcu102 | tee ${LOG}/target_zcu102.log
python -u target.py -t u280   | tee ${LOG}/target_u280.log
python -u target.py -t vck190 | tee ${LOG}/target_vck190.log

