# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:48:50 2020

@author: Administrator
"""
import numpy as np

''''training set'''

EPOCH_NUMS = 4000

WEIGHT_DECAY = 0.0002

MAX_TO_KEEP = 3

IMAGE_SIZE = 512

IMAGE_SIZE_Y = 512  

IMAGE_SIZE_X = 512

BATCH_SIZE = 1

LEARNING_RATE = 1e-4

VALIDATION_LEARNING_RATE = 0

OPTIC_LR = 0.00

DROPOUT = 1.0

TRAINING_CROSS_ITEMS = [0,1,2,3,4,5,6,7]

VALIDATION_CROSS_ITEMS = [8]

TEST_CROSS_ITEMS = [9]

TEST_EXTERNEL_ITEMS = [9]

INTERUPT_INDEX = 2833

''''ENABLE'''

PREPROCESS_ENABLE = False

RANDOM_FLIP_LR_ENABLE = True

RANDOM_FLIP_UD_ENABLE = True

RANDOM_ROTATE_ENABLE = True

RANDOM_CROP_ENABLE = True

ROTATE_MAX_ANGLE = 20

CROP_LEAST_COVER_PERCENTAGE = 1

PIC_SHOW = False

GAMMA_ENABLE = False

GENERATOR_ENABLE = True

OPTIC_ENABLE = True

BUILD_ENABLE = True

OPTICAL_MODEL_TRAINING_ENABLE = False

''''optics parameters'''

refractive_idcs = np.array([1.53770, 1.53116, 1.52750])

wave_lengths = np.array([486, 587, 656]) * 1e-9

PHASE_NUM = 90

patch_size = 512

sampling_interval = 5.86e-6   #5.86e-6 # pixel size /m

wave_resolution =512, 512 #2000, 2000# 512, 512 #853, 853

sensor_distance = 87.36e-3

obj_distance = 50e-3

focal_length = [31.6516e-3, 31.7917e-3, 32.0356e-3]

back_distance = 10e-3
