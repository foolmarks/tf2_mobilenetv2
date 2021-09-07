'''
 Copyright 2020 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Quantize the floating-point model
'''

'''
Author: Mark Harvey
'''


import argparse
import os
import shutil
import sys

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from dataset_utils import input_fn_test, input_fn_quant

DIVIDER = '-----------------------------------------'



def quant_model(build_dir,batchsize,evaluate):

    float_model = build_dir + '/float_model/f_model.h5'
    quant_model = build_dir + '/quant_model/q_model.h5'
    tfrec_dir = build_dir + '/tfrecords'


    '''
    Quantize the floating-point model
    Save to HDF5 file
    '''

    # make folder for saving quantized model
    head_tail = os.path.split(quant_model) 
    os.makedirs(head_tail[0], exist_ok = True)

    # load the floating point trained model
    float_model = load_model(float_model)

    # get input dimensions of the floating-point model
    height = float_model.input_shape[1]
    width = float_model.input_shape[2]

    # make TFRecord dataset and image processing pipeline
    quant_dataset = input_fn_quant(tfrec_dir, batchsize, height, width)

    # run quantization
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=quant_dataset)

    # saved quantized model
    quantized_model.save(quant_model)
    print('Saved quantized model to',quant_model)


    if (evaluate):
        '''
        Evaluate quantized model
        '''
        print('\n'+DIVIDER)
        print ('Evaluating quantized model..')
        print(DIVIDER+'\n')

        test_dataset = input_fn_test(tfrec_dir, batchsize, height, width)

        quantized_model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                                metrics=['accuracy'])

        scores = quantized_model.evaluate(test_dataset)

        print('Quantized model accuracy: {0:.4f}'.format(scores[1]*100),'%')
        print('\n'+DIVIDER)

    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-bd','--build_dir',  type=str, default='build', help='Build folder path. Default is build.')
    ap.add_argument('-b', '--batchsize',  type=int, default=50,      help='Batchsize for quantization. Default is 50')
    ap.add_argument('-e', '--evaluate',   action='store_true', help='Evaluate floating-point model if set. Default is no evaluation.')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --build_dir  : ', args.build_dir)
    print (' --batchsize  : ', args.batchsize)
    print (' --evaluate   : ', args.evaluate)
    print('------------------------------------\n')


    quant_model(args.build_dir, args.batchsize, args.evaluate)


if __name__ ==  "__main__":
    main()
