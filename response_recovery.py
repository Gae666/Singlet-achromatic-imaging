 # -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:21:48 2020

@author: Administrator

"""

import sys
sys.dont_write_bytecode = True

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import pickle
import numpy as np
import tensorflow as tf
from main_function import MainFunction
from parameters import *
from datasets_processing import TrainingData
import warnings
warnings.filterwarnings('ignore')

tf.reset_default_graph()

def train():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_function = MainFunction(sess)
        ckpt_num = 0
        if ckpt_num == 0:
            sess.run(tf.global_variables_initializer())
        else:
            ckpt_path = 'all_results/ckpt_result/' + str(ckpt_num) + '.ckpt'
            train_function.restore(ckpt_path)
        train_function.train(EPOCH_NUMS, ckpt_num)

def test_external():
    #config = tf.ConfigProto(log_device_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        test_externel_function = MainFunction(sess)
        ckpt_path = 'all_results/ckpt_result/3526.ckpt'
        test_externel_function.restore(ckpt_path)
        test_externel_function.test_externel()
def validation_external():
    #config = tf.ConfigProto(log_device_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        validation_external_function = MainFunction(sess)
        ckpt_path = 'all_results/ckpt_result/454.ckpt'
        validation_external_function.restore(ckpt_path)
        validation_external_function.validation_external()    
def interupt():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        interupt_function = MainFunction(sess)
        ckpt_path = 'all_results/ckpt_result/2833.ckpt'
        interupt_function.restore(ckpt_path)
        interupt_function.train(EPOCH_NUMS,2833)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please tell me more about the function you want, such as test or train')
        exit(-1)

    function = sys.argv[1]
    globals()[function](*sys.argv[2:])

