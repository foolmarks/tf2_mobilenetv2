#!/bin/sh

# Author: Mark Harvey

if [ $1 = zcu102 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
      TARGET=zcu102
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU102.."
      echo "-----------------------------------------"
elif [ $1 = zcu104 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
      TARGET=zcu104
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU104.."
      echo "-----------------------------------------"
elif [ $1 = u280 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8L/U280/arch.json
      TARGET=u280
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ALVEO U280.."
      echo "-----------------------------------------"
elif [ $1 = vck190 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json
      TARGET=vck190
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR VERSAL VCK190.."
      echo "-----------------------------------------"
else
      echo  "Target not found. Valid choices are: zcu102, zcu104, u280, vck190 ..exiting"
      exit 1
fi

BUILD=$2
LOG=$3

compile() {
      vai_c_tensorflow2 \
            --model           ${BUILD}/quant_model/q_model.h5 \
            --arch            $ARCH \
            --output_dir      ${BUILD}/compiled_model_${TARGET} \
            --net_name        mobilenetv2
}


compile 2>&1 | tee ${LOG}/compile_${TARGET}.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"
