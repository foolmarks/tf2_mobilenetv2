
'''
Make the target folder
Creates images, copies application code and compiled xmodel in a single folder
'''

'''
Author: Mark Harvey
'''


import argparse
import os
import shutil
import sys
import cv2
from tqdm import tqdm

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf

from dataset_utils import input_fn_image

DIVIDER = '-----------------------------------------'


def make_target(build_dir,target,num_images,input_height,input_width,app_dir):

    tfrec_dir = build_dir + '/tfrecords'
    target_dir = build_dir + '/target_' + target
    image_dir = target_dir + '/images'
    model = build_dir + '/compiled_model_' + target + '/mobilenetv2.xmodel'


    # remove any previous data
    shutil.rmtree(target_dir, ignore_errors=True)    
    os.makedirs(target_dir)
    os.makedirs(image_dir)

    # make the dataset
    target_dataset = input_fn_image(tfrec_dir,1,input_height,input_width)

    '''
    # extract images & labels from TFRecords
    # save as JPEG image files
    '''
    i = 0
    for tfr in tqdm(target_dataset):

        label = tfr[1][0].numpy()

        # reshape image to remove batch dimension
        img = tf.reshape(tfr[0], [tfr[0].shape[1],tfr[0].shape[2],tfr[0].shape[3]] )

        # JPEG encode
        img = tf.cast(img, tf.uint8)
        img = tf.io.encode_jpeg(img)

        # save as file
        filepath =  os.path.join(image_dir, str(label)+'_image'+str(i)+'.jpg' )
        tf.io.write_file(filepath, img)

        i += 1

        if i==num_images:
          break



    # copy application code
    print('Copying application code from',app_dir,'...')
    shutil.copy(os.path.join(app_dir, 'app_mt.py'), target_dir)

    # copy compiled model
    print('Copying compiled model from',model,'...')
    shutil.copy(model, target_dir)


    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-bd','--build_dir',   type=str,  default='build',  help='Build folder path. Default is build.')
    ap.add_argument('-t', '--target',      type=str,  default='zcu102', help='Name of target device or board. Default is zcu102.')
    ap.add_argument('-n', '--num_images',  type=int,  default=1000,          help='Number of test images. Default is 1000')
    ap.add_argument('-ih','--input_height',type=int,  default=224,           help='Input image height in pixels.')
    ap.add_argument('-iw','--input_width', type=int,  default=224,           help='Input image width in pixels.')
    ap.add_argument('-a', '--app_dir',     type=str,  default='application', help='Full path of application code folder. Default is application')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --build_dir    : ', args.build_dir)
    print (' --target       : ', args.target)
    print (' --num_images   : ', args.num_images)
    print (' --input_height : ', args.input_height)
    print (' --input_width  : ', args.input_width)
    print (' --app_dir      : ', args.app_dir)
    print('------------------------------------\n')


    make_target(args.build_dir,args.target,args.num_images,args.input_height,args.input_width,args.app_dir)


if __name__ ==  "__main__":
    main()
  
