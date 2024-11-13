# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:23:01 2020

@author: Administrator
"""
import pickle
import random
from parameters import *
from datasets_processing import TrainingData
import tensorflow as tf
import matplotlib.pyplot as plt

import math

def load_data(items):
    data = []
    
    for item in items:
        data_item_dir = '/media/aa/DATA2/HC/test_' + str(item) +'.pkl'
        with open(data_item_dir, 'rb') as file_object:
            data = data + pickle.load(file_object)
    return data

def preprocess(image):
    if PREPROCESS_ENABLE:
        height = image.shape[0]
        width = image.shape[1]
        if RANDOM_FLIP_LR_ENABLE:
            image = tf.image.random_flip_left_right(image)
        if RANDOM_FLIP_UD_ENABLE:
            image = tf.image.random_flip_up_down(image)
        if RANDOM_ROTATE_ENABLE:
            image = tf.contrib.image.rotate(image, (random.random() - 0.5) * math.radians(ROTATE_MAX_ANGLE))
        if RANDOM_CROP_ENABLE:
            xmax = 1 - random.random()*(1 - CROP_LEAST_COVER_PERCENTAGE)
            ymax = 1 - random.random() *(1 - CROP_LEAST_COVER_PERCENTAGE)
            start = max(0, xmax - ymax * height / width)
            end = ymax - CROP_LEAST_COVER_PERCENTAGE
            xmin = start + random.random() * (end - start)
            ymin = ymax - (xmax - xmin) * width / height
            image = tf.expand_dims(image, 0)       
            image = tf.image.crop_and_resize(image,[[ymin,xmin,ymax,xmax]],box_ind=[0],crop_size=(512,512))            
            
    return image

class DataGenerator:
    '''训练数据生成''' 
    def __init__(self, is_training, items, batch_size):
        self.is_training = is_training
        self.data = load_data(items)
        random.shuffle(self.data)
        self.batch_size = batch_size
        self.data_nums = len(self.data)
        self.count = 0
        self.ground_truth = []
        self.simulate_image = []  
        self.filename = []
        for i in range(self.data_nums):
            self.ground_truth.append(self.data[i].ground_truth)
            self.simulate_image.append(self.data[i].simulate_image)
            self.filename.append(self.data[i].filename)
        
    def get_one_batch(self):
        '''获取一个batch的数据'''
        one_batch = [[], [], []]
        if self.count + self.batch_size >= self.data_nums:
            self.count = 0
            random.shuffle(self.data)
            print('shuffle done')
        for i in range(self.count, self.count + self.batch_size):
            one_batch[0].append(self.simulate_image[i])
            one_batch[1].append(self.ground_truth[i])
            one_batch[2].append(self.filename[i])
        self.count = self.count + self.batch_size
        return one_batch
        
                
                
            
