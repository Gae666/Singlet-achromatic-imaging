 #-*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:45:39 2020

@author: Administrator
"""
from parameters import *
import tensorflow as tf
from data_generator import load_data
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from data_generator import DataGenerator
from sub_files import Tee
import matplotlib.pyplot as plt

import random
import math
import warnings
import os
import layers.optics as optics
import layers.deconv as deconv
import poppy
import matplotlib.pyplot as plt

if not os.path.exists('all_results/mat'):
    os.makedirs('all_results/mat')

warnings.filterwarnings('ignore')

slim = tf.contrib.slim
reg = tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)


def weight_init(name, shape):
    weight = tf.get_variable(name=name + '_filter_', shape=shape,
                             initializer=tf.contrib.layers.variance_scaling_initializer(), regularizer=reg)
    return weight


def bias_init(name, shape):
    bias = tf.get_variable(name=name + '_bias_', shape=shape, initializer=tf.constant_initializer(0.0))
    return bias


def multi_conv(layer_input, name, shape):
    weight = weight_init(name, shape)
    bias = bias_init(name, shape[3])
    result = tf.nn.conv2d(layer_input, weight, strides=[1, 1, 1, 1], padding='SAME')
    result = tf.nn.bias_add(result, bias)
    return result


def multi_scale(layer_input, name, small_channel, medium_channel, long_channel):
    mult = {}
    shape = layer_input.get_shape()

    mult['small_conv'] = multi_conv(layer_input, name + '_small', [1, 1, shape[3], small_channel])
    mult['medium_conv'] = multi_conv(layer_input, name + '_medium', [3, 3, shape[3], medium_channel])
    mult['long_conv'] = multi_conv(layer_input, name + '_long', [5, 5, shape[3], long_channel])
    mult['concat'] = tf.concat([mult['small_conv'], mult['medium_conv'], mult['long_conv']], 3)
    mult['relu'] = tf.nn.relu(mult['concat'])
    return mult['relu']


'''percptual loss'''
def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.max_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias

vgg_path=scipy.io.loadmat('vgg19')
print("[i] Loaded pre-trained vgg19 parameters")
# build VGG19 to load pre-trained parameters
def build_vgg19(input):
    net={}
    vgg_layers=vgg_path['layers'][0]
    net['input']=input
    net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
    net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
    net['pool1']=build_net('pool',net['conv1_2'])
    net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
    net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
    net['pool2']=build_net('pool',net['conv2_2'])
    net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
    net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
    net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
    net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
    net['pool3']=build_net('pool',net['conv3_4'])
    net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
    net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
    net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
    net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
    net['pool4']=build_net('pool',net['conv4_4'])
    net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
    net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
    return net

def compute_l1_loss(input, output):
    return tf.reduce_mean(tf.abs(input-output))


def compute_l2_loss(input, output):
    return tf.losses.mean_squared_error(input, output)

def compute_percep_loss(input, output):
    input = input[:, :, :, ::-1]*255.0
    output = output[:, :, :, ::-1]*255.0
    sub_mean = tf.constant([104.0, 117.0, 123.0],dtype=tf.float32,shape=[1, 1, 1, 3],name='img_sub_mean')
    input = input - sub_mean
    output = output - sub_mean
    vgg_real=build_vgg19(output)
    vgg_fake=build_vgg19(input)
    p0=compute_l2_loss(vgg_real['input'],vgg_fake['input'])/21438
    p1=compute_l2_loss(vgg_real['conv1_2'],vgg_fake['conv1_2'])/29928
    p2=compute_l2_loss(vgg_real['conv2_2'],vgg_fake['conv2_2'])/49163
    p3=compute_l2_loss(vgg_real['conv3_2'],vgg_fake['conv3_2'])/52520
    p4=compute_l2_loss(vgg_real['conv4_2'],vgg_fake['conv4_2'])/34523/10.0
    p5=compute_l2_loss(vgg_real['conv5_2'],vgg_fake['conv5_2'])/21552*10.0
    return p0+p1+p2+p3+p4+p5

def input_crop(image,image_size, crop_least_cover_percentage, batch_size):
    boxes = []
    for box_count in range(batch_size):
        xmax = 1 - random.random() * (1  - crop_least_cover_percentage)
        ymax = 1 - random.random() * (1 - crop_least_cover_percentage)
        start = max(0, xmax - ymax * image_size / image_size)
        end = ymax - crop_least_cover_percentage
        xmin = start + random.random() * (end - start)
        ymin = ymax - (xmax - xmin) * image_size / image_size
        boxes.append([ymin, xmin, ymax, xmax])
    edit_image = tf.image.crop_and_resize(image, boxes, box_ind=list(range(0, batch_size)),
                                           crop_size=(image_size, image_size))
    return edit_image



class MainFunction:
    '''训练及测试函数主体'''

    def __init__(self, sess=None):
        self.sess = sess

        self.dropout = tf.placeholder(tf.float32, shape=(), name='dropout')
        #self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(LEARNING_RATE,self.global_step,135000,0.4,staircase=True)
        self.ground_truth = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3), name='ground_truth')
        self.reconstructed_images = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3),
                                                   name='reconstructed_images')
        self.is_training = tf.placeholder(tf.bool)
        images = tf.sqrt(self.reconstructed_images)

        
        '''进行Gamma变换'''
        if GAMMA_ENABLE:
            images = self.reconstructed_images * 1.0 / 255
            images = (tf.pow(images, 0.4545) * 255.0)
            
        if GENERATOR_ENABLE and OPTIC_ENABLE and BUILD_ENABLE:
            
            with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY)):
                    self.complex= self.generator(images, PHASE_NUM, self.dropout, self.is_training)
                    print('complex number is ', len(self.complex))
                  

                    self.optics_image = tf.zeros_like(tf.cast(self.complex[0], tf.float32))
                    self.sensor_incident_image = tf.zeros_like(tf.cast(self.complex[0], tf.complex64))
                    self.wave_all = []
                    self.incident_wave_all = []
                   
                    with tf.variable_scope("optical_model", reuse=False):
                        self.wave = self.complex[0]
                        #self.optics_sensor_image, self.psf, self.lens, self.target_depth, self.intensity_psf, self.focal = self.optics_model(self.wave)
                        self.optics_sensor_image, self.psf, self.intensity_psf, self.sensor_incident_field = self.optics_model(self.wave)
                        self.optics_image = self.optics_sensor_image + self.optics_image
                        self.incident_wave_all.append(self.sensor_incident_field)
                    with tf.variable_scope("optical_model", reuse=True):
                        for i in range(0, int(PHASE_NUM/6)):
                            self.wave = self.complex[i]
                            #self.optics_sensor_image, self.psf, self.lens, self.target_depth, self.intensity_psf, self.focal = self.optics_model(self.wave)
                            self.optics_sensor_image, self.psf, self.intensity_psf, self.sensor_incident_field = self.optics_model(self.wave)
                            self.optics_image = self.optics_sensor_image + self.optics_image
                            self.sensor_incident_image = self.sensor_incident_field + self.sensor_incident_image
                            self.wave_all.append(self.optics_sensor_image)
                            self.incident_wave_all.append(self.sensor_incident_field)
    
                    
                    self.conv10 = self.optics_image 
                    self.sensor_incident_image = self.sensor_incident_image# sum wave_alll 
                    
       
          
        '''loss'''
        self.MSE_loss = 6*(compute_l2_loss(self.conv10, self.ground_truth))
        self.MAE_loss = 6*(compute_l1_loss(self.conv10, self.ground_truth))
        self.percep_loss = compute_percep_loss(self.conv10, self.ground_truth)
        self.SSIM_loss = 0*0.01*(1 - tf.reduce_mean(tf.image.ssim(self.conv10, self.ground_truth,max_val=1.0)))
        self.loss = self.MSE_loss + self.MAE_loss + self.percep_loss + self.SSIM_loss

        scalar_summaries = []
        scalar_summaries.append(tf.summary.scalar('loss', self.loss))
        self.scalar_summaries = tf.summary.merge(scalar_summaries)
        if OPTIC_ENABLE and GENERATOR_ENABLE and BUILD_ENABLE:
            loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        else:
            loss_reg = 0 
        self.total_loss = self.loss + loss_reg
        self.Optical_Variables =tf.get_collection('Optical_Variables')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops):
            if OPTICAL_MODEL_TRAINING_ENABLE:
                self.train_step_adam_optics = tf.train.AdamOptimizer(OPTIC_LR).minimize(self.total_loss,
                                                                                       var_list=self.Optical_Variables() + optimizer.variables())
            self.train_step_adam = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss,global_step = self.global_step)
        tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)
        
        
        '''对图像进行预处理'''
        self.raw_images_blur = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3), name='raw_images_blur')
        self.raw_images_clear = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3), name='raw_images_clear')
        edit_images_blur = self.raw_images_blur
        edit_images_clear = self.raw_images_clear

        if PREPROCESS_ENABLE:
            if RANDOM_ROTATE_ENABLE:
                random_angles = tf.random.uniform(shape=(tf.shape(edit_images_blur)[0],),
                                                  minval=-ROTATE_MAX_ANGLE * np.pi / 180,
                                                  maxval=ROTATE_MAX_ANGLE * np.pi / 180)

                edit_images_blur = tf.contrib.image.transform(edit_images_blur,tf.contrib.image.angles_to_projective_transforms(random_angles, tf.cast(tf.shape(edit_images_blur)[1], tf.float32), tf.cast(tf.shape(edit_images_blur)[2], tf.float32)))
                edit_images_clear = tf.contrib.image.transform(edit_images_clear,tf.contrib.image.angles_to_projective_transforms(random_angles, tf.cast(tf.shape(edit_images_clear)[1], tf.float32), tf.cast(tf.shape(edit_images_clear)[2], tf.float32)))
            if RANDOM_CROP_ENABLE:
                boxes = []
                for box_count in range(BATCH_SIZE):
                    xmax = 1 - random.random() * (1 - CROP_LEAST_COVER_PERCENTAGE)
                    ymax = 1 - random.random() * (1 - CROP_LEAST_COVER_PERCENTAGE)
                    start = max(0, xmax - ymax * IMAGE_SIZE / IMAGE_SIZE)
                    end = ymax - CROP_LEAST_COVER_PERCENTAGE
                    xmin = start + random.random() * (end - start)
                    ymin = ymax - (xmax - xmin) * IMAGE_SIZE / IMAGE_SIZE
                    boxes.append([ymin, xmin, ymax, xmax])
                edit_images_blur = tf.image.crop_and_resize(edit_images_blur, boxes, box_ind=list(range(0, BATCH_SIZE)),
                                                       crop_size=(IMAGE_SIZE_X, IMAGE_SIZE_Y))
                edit_images_clear = tf.image.crop_and_resize(edit_images_clear, boxes, box_ind=list(range(0, BATCH_SIZE)),
                                                       crop_size=(IMAGE_SIZE_X, IMAGE_SIZE_Y))                                                       
        self.edit_images_blur = edit_images_blur
        self.edit_images_clear = edit_images_clear
        
    def bias_init_G(self, name, shape):
        bias = tf.get_variable(name=name + '_bias_', shape=shape, initializer=tf.constant_initializer(0.0))
        if (bias not in tf.get_collection('G_Variables')):
            tf.add_to_collection('G_Variables',bias)
        return bias
    
    def weight_init_G(self, name, shape):
        weight = tf.get_variable(name=name + '_filter_', shape=shape,
                             initializer=tf.variance_scaling_initializer(), regularizer=reg)
        if (weight not in tf.get_collection('G_Variables')):
            tf.add_to_collection('G_Variables',weight)
        return weight
    
    #def symmetric_padding(inputs, padding):
       # batch_size, height, width, channels = inputs.shape.as_list()
       # paddings 
    
    
    
    def channel_attation(self,input_block,weight_name):
        w = input_block.get_shape()[1]
        h = input_block.get_shape()[2]
        channel_num = input_block.get_shape()[3]
        
        Q = tf.nn.conv2d(input_block, self.weight_init_G('channel_conv1' + weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        Q = tf.nn.bias_add(Q, self.bias_init_G('channel_bias1' + weight_name, [channel_num]))
        Q = tf.nn.relu(Q)
        K = tf.nn.conv2d(input_block, self.weight_init_G('channel_conv2'+ weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        K = tf.nn.bias_add(K, self.bias_init_G('channel_bias2' + weight_name, [channel_num]))
        K = tf.nn.relu(K)
        V = tf.nn.conv2d(input_block, self.weight_init_G('channel_conv3'+ weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        V = tf.nn.bias_add(V, self.bias_init_G('channel_bias3' + weight_name, [channel_num]))
        V = tf.nn.relu(V)
        
        
        Q = tf.reshape(Q,[-1, w * h, channel_num])
        Q = tf.transpose(Q, [0,2,1])
        K = tf.reshape(K,[-1, w * h, channel_num])
        V = tf.reshape(V,[-1, w * h, channel_num])
        V = tf.transpose(V, [0,2,1])
        print('The shape of Q is',Q.get_shape())
        
        channel_weight = tf.nn.softmax(tf.matmul(Q,K))
        print('The shpae of channel_weight is', channel_weight.get_shape())
        
        weighted_block = tf.matmul(channel_weight,V)
        weighted_block = tf.transpose(weighted_block)
        weighted_block = tf.reshape(weighted_block, [-1, w, h, channel_num])
        print('The shape of weighted_block is',weighted_block.get_shape())
        
        channel_attation_block = (input_block + weighted_block)
        return channel_attation_block
    
    def spatial_attation(self,input_block,weight_name):
        w = input_block.get_shape()[1]
        h = input_block.get_shape()[2]
        channel_num = input_block.get_shape()[3]
        
        Q = tf.nn.conv2d(input_block, self.weight_init_G('spatial_conv1' + weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        Q = tf.nn.bias_add(Q, self.bias_init_G('spatial_bias1' + weight_name, [channel_num]))
        Q = tf.nn.relu(Q)
        K = tf.nn.conv2d(input_block, self.weight_init_G('spatial_conv2'+ weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        K = tf.nn.bias_add(K, self.bias_init_G('spatial_bias2' + weight_name, [channel_num]))
        K = tf.nn.relu(K)
        V = tf.nn.conv2d(input_block, self.weight_init_G('spatial_conv3'+ weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        V = tf.nn.bias_add(V, self.bias_init_G('spatial_bias3' + weight_name, [channel_num]))
        V = tf.nn.relu(V)
        
        Q = tf.reshape(Q,[-1, w * h, channel_num])
        K = tf.reshape(K,[-1, w * h, channel_num])
        K = tf.transpose(K, [0,2,1])
        V = tf.reshape(V,[-1, w * h, channel_num])
        V = tf.transpose(V, [0,2,1])
        print('The shape of K is',K.get_shape())
        
        spatial_weight = tf.nn.softmax(tf.matmul(Q,K))
        spatial_weight = tf.transpose(spatial_weight, [0,2,1])
        print('The shpae of spatial_weight is', spatial_weight.get_shape())
        
        weighted_block = tf.matmul(V,spatial_weight)
        weighted_block = tf.transpose(weighted_block)
        weighted_block = tf.reshape(weighted_block,[-1, w, h, channel_num])
        print('The shape of weighted_block is',weighted_block.get_shape())
        
        spatial_attation_block = (input_block + weighted_block)
        return spatial_attation_block
    
        
    def optics_model(self, image):
        '''image: complex wavefront'''
        # creat zernike polynomial 231 terms
        if not os.path.exists('zernike_volume_%d.npy'%wave_resolution[0]):
            zernike_volume = optics.get_zernike_volume(resolution=wave_resolution[0], n_terms=37).astype(np.float32)
            np.save('zernike_volume_%d.npy'%wave_resolution[0], zernike_volume)
        else:
            zernike_volume = np.load('zernike_volume_%d.npy' % wave_resolution[0])
        input_img = image

        
       
       
        '''simulate different depths' psf '''
        all_depths = tf.convert_to_tensor([1 / 2, 1 / 1.5, 1 / 1, 1 / 0.5, 1000], tf.float32)

        depth_bins = []
        for i in range(3):
            depth_idx = tf.multinomial(tf.log([5 * [1 / 5]]), num_samples=1)
            depth_bins.append(all_depths[tf.cast(depth_idx[0][0], tf.int32)])
            
        test_depth = np.concatenate(
            [np.ones((patch_size // len(depth_bins), patch_size)) * i for i in range(len(depth_bins))], axis=0)[:, :, None]

        depth_map = np.expand_dims(test_depth, axis = 0)
        target_depth_initializer = tf.constant_initializer(obj_distance)
        target_depth = tf.get_variable(name="target_depth",
                                           shape=(),
                                           dtype=tf.float32,
                                           trainable=False,
                                           initializer=target_depth_initializer)
        target_depth = tf.abs(target_depth) # Enforce that depth is positive.
        optical_system = optics.ZernikeSystem(zernike_volume=zernike_volume,
                    back_distance = back_distance,
                    target_distance=target_depth, # object distance /meter
                    wave_resolution=wave_resolution,
                    upsample=False, #image resolution does not match the wave resolution
                    wave_lengths=wave_lengths,
                    sensor_resolution=(patch_size, patch_size),# image resolution
                    height_tolerance=0, # lens fabrication error
                    refractive_idcs=refractive_idcs,
                    input_sample_interval=sampling_interval, #pixel size
                    sensor_distance=sensor_distance, #image distance
                    depth_bins=depth_bins,
                    focal_length=focal_length) # object distance

            # We want to be robust to the noise level. Thus we pick a noise level at random.
        # noise_sigma = tf.random_uniform(minval=0.001, maxval=0.02, shape=[])
        # gaussian noise_sigma = 0
        sensor_img = optical_system.get_zemax_img(input_img=input_img)
        output_image = tf.cast(sensor_img, tf.float32) # 目前只是wave[i]的一片 
        psf_tar = optical_system.psf
        #lens = optical_system.height_map
        intensity_psf = optical_system.intensity_psf
        sensor_incident_field = optical_system.sensor_incident_field
        #focal = optical_system.focal
        '''
            # Now deconvolve
        pad_width = output_image.shape.as_list()[1] // 2

        output_image = tf.pad(output_image, [[0, 0], [pad_width, pad_width], [pad_width, pad_width], [0, 0]])
        output_image = deconv.inverse_filter(output_image, output_image, optical_system.target_psf,
                                                 init_gamma=0)
        output_image = output_image[:, pad_width:-pad_width, pad_width:-pad_width, :]
        '''
        return output_image, psf_tar,  intensity_psf, sensor_incident_field #lens, target_depth, intensity_psf, focal 
    '''build the generator to reconstruct complex wavefront from the input intensity objective distribution'''
    
    def generator(self, images, phase_num, dropout, is_training):
        # images = input_crop(images, 1356, 1, 1)
        #complex_wave = tf.cast(tf.sqrt(images), tf.complex64)
        conv1 = slim.conv2d(images, 32, [3, 3], activation_fn = None)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        conv1 = tf.nn.relu(conv1)
        conv1 = slim.conv2d(conv1, 32, [3, 3], activation_fn = None)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        conv1 = tf.nn.relu(conv1)
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

        conv2 = slim.conv2d(pool1, 64, [3, 3], activation_fn = None)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        conv2 = tf.nn.relu(conv2)
        conv2 = slim.conv2d(conv2, 64, [3, 3], activation_fn = None)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        conv2 = tf.nn.relu(conv2)
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

        conv3 = slim.conv2d(pool2, 128, [3, 3], activation_fn = None)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        conv3 = tf.nn.relu(conv3)
        conv3 = slim.conv2d(conv3, 128, [3, 3], activation_fn = None)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        conv3 = tf.nn.relu(conv3)
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

        conv4 = slim.conv2d(pool3, 256, [3, 3], activation_fn = None)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training)
        conv4 = tf.nn.relu(conv4)
        conv4 = slim.conv2d(conv4, 256, [3, 3], activation_fn = None)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training)
        conv4 = tf.nn.relu(conv4)
        pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

        conv5 = slim.conv2d(pool4, 512, [3, 3], activation_fn = None)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training)
        conv5 = tf.nn.relu(conv5)
        conv5 = slim.conv2d(conv5, 512, [3, 3], activation_fn = None)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training)
        conv5 = tf.nn.relu(conv5)

        conv5 = self.channel_attation(conv5, 'generator_channel')

        up6 = self.upsample_and_concat(conv5, conv4, 256, 512)
        conv6 = tf.layers.batch_normalization(up6, training=is_training)
        conv6 = slim.conv2d(conv6, 256, [3, 3], activation_fn = None)
        conv6 = tf.nn.relu(conv6)
        conv6 = slim.conv2d(conv6, 256, [3, 3], activation_fn = None)
        conv6 = tf.nn.relu(conv6)

        up7 = self.upsample_and_concat(conv6, conv3, 128, 256)
        conv7 = tf.layers.batch_normalization(up7, training=is_training)
        conv7 = slim.conv2d(conv7, 128, [3, 3], activation_fn = None)
        conv7 = tf.nn.relu(conv7)
        conv7 = slim.conv2d(conv7, 128, [3, 3], activation_fn = None)
        conv7 = tf.nn.relu(conv7)

        up8 = self.upsample_and_concat(conv7, conv2, 64, 128)
        conv8 = tf.layers.batch_normalization(up8, training=is_training)
        conv8 = slim.conv2d(conv8, 64, [3, 3], activation_fn = None)
        conv8 = tf.nn.relu(conv8)
        conv8 = slim.conv2d(conv8, 64, [3, 3], activation_fn = None)
        conv8 = tf.nn.relu(conv8)

        up9 = self.upsample_and_concat(conv8, conv1, 32, 64)
        conv9 = tf.layers.batch_normalization(up9, training=is_training)
        conv9 = slim.conv2d(conv9, 32, [3, 3], activation_fn = None)
        conv9 = tf.nn.relu(conv9)
        conv9 = slim.conv2d(conv9, 32, [3, 3], activation_fn = None)
        conv9 = tf.nn.relu(conv9)
        print("conv9.shape:{}".format(conv9.get_shape()))
        
        conv9 = tf.nn.dropout(conv9, dropout)        
        conv10 = slim.conv2d(conv9, phase_num * 2, [1, 1], activation_fn = None) # phase
        #conv10 = np.pi*tf.nn.tanh(conv10)
        phase_and_amp = conv10
        print("phase_and_amp.shape:{}".format(conv10.get_shape()))
        #phase = np.pi * tf.nn.tanh(phase)
        # phase = input_crop(conv10, 1356, 1, 1)  
        # images = input_crop(images, 1356, 1, 1)
        complex_waves = self.coherent_propogate(phase_and_amp, phase_num)
        '''complex wavefront'''
        # images = tf.cast(tf.sqrt(images), tf.complex64)
        # phase = optics.compl_exp_tf(conv10,dtype=tf.complex64)
        # complex_wave = tf.multiply(images,phase)
        
        return complex_waves
    
    def coherent_propogate(self, phase_and_amp, phase_num):
        #image = tf.pow(image, 2.2)
        #image = tf.cast(image, tf.complex64)
        complex_waves = []
        weight = tf.get_variable('pahse_weight', shape=[phase_num/3,], initializer=tf.constant_initializer((1/phase_num)*0))
        weight = tf.clip_by_value(weight, 0, 1)
        #weight = tf.cast(weight, tf.complex64)
        for i in range(0, int(phase_num/6)):
            phase_and_amp_index = phase_and_amp[:,:,:,6*i:6*i+6]
            phase = phase_and_amp_index[:,:,:,0:3]
            amp = tf.cast(tf.abs(phase_and_amp_index[:,:,:,3:6]), tf.complex64)
            phase = optics.compl_exp_tf(phase, dtype=tf.complex64)
            complex_wave = tf.multiply(amp, phase)
            complex_waves.append(complex_wave)
        return complex_waves
    
  

    def upsample_and_concat(self,x1, x2, output_channels, in_channels):
        pool_size = 2
        deconv_filter = tf.Variable(
            tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

        deconv_output = tf.concat([deconv, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])
        return deconv_output

    def restore(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)



    def validation_external(self):
        batch_size = BATCH_SIZE
        validation_data_generating = DataGenerator(True, VALIDATION_CROSS_ITEMS, batch_size)
        validation_batch_nums = validation_data_generating.data_nums // batch_size
           
        test_loss_sum = 0
        test_constructed_spectrums_all = []
        test_sensor_incident_field_all = []
        test_ground_truth_all = []
        test_filenames_all = []
        test_mat_count = 0
        test_images_all = []
        test_lens_all = []
        test_target_distance_all = []
        
        validation_loss_sum = 0
        validation_MSE_loss_sum = 0
        validation_percep_loss_sum = 0
        validation_mat_count = 0
          
        for jj in range(validation_batch_nums): 
            validation_one_batch = validation_data_generating.get_one_batch()
            edit_images_blur, edit_images_clear = self.sess.run([self.edit_images_blur, self.edit_images_clear],
                                                                    feed_dict={self.raw_images_blur: np.stack(validation_one_batch[0]), self.raw_images_clear: np.stack(validation_one_batch[1])})
                      
            validation_loss,_, validation_MSE_loss, validation_MAE_loss, validation_percep_loss,  SSIM_loss, validation_constructed_spectrums, psf, intensity_psf, incident_wave_all, wave_all, validation_sensor_incident_image = self.sess.run(
                    [
                     self.total_loss,
                     self.train_step_adam,
                     self.MAE_loss,
                     self.MSE_loss,
                     self.percep_loss,
                     #self.target_depth,
                     self.SSIM_loss,
                     self.conv10,
                     self.psf,
                    # self.lens,
                     self.intensity_psf,
                     self.incident_wave_all,
                    # self.focal,
                     self.wave_all,
                     self.sensor_incident_image
                     ],
                    feed_dict=
                    {self.dropout: DROPOUT,
                     self.learning_rate: VALIDATION_LEARNING_RATE,
                     self.reconstructed_images: np.stack(edit_images_blur),
                     self.ground_truth: np.stack(edit_images_clear),
                     self.is_training: True
                     })
                
            if validation_mat_count == 0:
                validation_constructed_spectrums_all = validation_constructed_spectrums
                validation_sensor_incident_image_all = validation_sensor_incident_image
                validation_input_images_all = edit_images_blur
                validation_ground_truth_all = edit_images_clear
                      
            else:
                validation_constructed_spectrums_all = np.concatenate(
                            [validation_constructed_spectrums_all, validation_constructed_spectrums], axis=0)
                validation_sensor_incident_image_all = np.concatenate(
                          [validation_sensor_incident_image_all, validation_sensor_incident_image], axis=0)
                validation_input_images_all = np.concatenate(
                            [validation_input_images_all, edit_images_blur], axis=0)
                validation_ground_truth_all = np.concatenate(
                            [validation_ground_truth_all, edit_images_clear], axis=0)
                       
                        
            validation_mat_count += 1
                    
            validation_loss_sum += validation_loss
            validation_MSE_loss_sum +=validation_MSE_loss
            validation_MAE_loss_sum +=validation_MAE_loss
            validation_percep_loss_sum += validation_percep_loss
                
            print('the', jj, 'in validation_batch_nums ', [validation_loss, validation_MSE_loss, validation_MAE_loss, validation_percep_loss, SSIM_loss ])
        validation_loss_ave = validation_loss_sum / validation_batch_nums
        validation_MSE_loss_ave = validation_MSE_loss_sum / validation_batch_nums
        validation_MAE_loss_ave = validation_MAE_loss_sum / validation_batch_nums
        validation_percep_loss_ave = validation_percep_loss_sum/ validation_batch_nums
        print('Validation: now the validation loss is: ', [validation_loss_ave,validation_MSE_loss_ave,validation_MAE_loss_ave,
                                                                             validation_percep_loss_ave])#target_depth
          
        scipy.io.savemat('all_results/mat/validation_external_454.mat',
                                        {'constructed_spectrums': validation_constructed_spectrums_all,
                                          # 'optics_images': optics_images_all,
                                          'input_images':validation_input_images_all,
                                          'ground_truths':validation_ground_truth_all,
                                          'intensity_psf':intensity_psf,
                                          'psfs': psf,
                                          'sensor_incident_image':validation_sensor_incident_image_all,
                                          #'lens': lens,
                                          #'obj_distance':target_depth,
                                          'wave_all':wave_all,
                                          'incident_wave_all':incident_wave_all,
                                          'focal_length':focal_length})
                                          #'complex_waves': complex_wave_all})
      


    def test_externel(self):
        '''parameter'''
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # 计算参数量
        total_parameters = 0
        for variable in tf.trainable_variables():
        
           # shape是一个数组，数组的长度代表了变量的维度数量
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters

        print(f"Total number of parameters: {total_parameters}")
        test_datas = load_data(TEST_CROSS_ITEMS)
        test_data = [[], [], []]
        test_count = 0
        test_collection = []
        for r in test_datas:
            test_data[0].append(r.simulate_image)
            test_data[1].append(r.ground_truth)
            test_data[2].append(r.filename)
            test_count += 1
            if test_count >= BATCH_SIZE:
                test_collection.append(test_data) 
                test_count = 0
                test_data = [[], [], []]
        if test_count != 0:
            test_collection.append(test_data)
        test_loss_sum = 0
        test_constructed_spectrums_all = []
        test_sensor_incident_field_all = []
        test_ground_truth_all = []
        test_filenames_all = []
        test_mat_count = 0
        test_images_all = []
        test_lens_all = []
        test_target_distance_all = []
        for test_bag in test_collection:
            test_loss, test_constructed_spectrums, psf, intensity_psf, incident_wave_all, wave_all, test_sensor_incident_field = self.sess.run(
                [self.total_loss,
                 self.conv10,
                 self.psf,
                 self.intensity_psf,
                 self.incident_wave_all,
              
                 self.wave_all,
                 self.sensor_incident_field
                # self.lens
                 ],
                feed_dict=
                {self.dropout: 1.0,
                 self.reconstructed_images: np.stack(test_bag[0]),
                 self.ground_truth: np.stack(test_bag[1]),
                 self.is_training: True
                 })
            if test_mat_count == 0:
                test_constructed_spectrums_all = test_constructed_spectrums
                test_ground_truth_all = np.stack(test_bag[1])
                test_filenames_all = np.stack(test_bag[2])
                test_images_all = np.stack(test_bag[0])
                #test_lens_all = lens
                test_sensor_incident_field_all = test_sensor_incident_field
                #test_target_distance_all.append(target_distance)
            else:
                test_constructed_spectrums_all = np.concatenate(
                    [test_constructed_spectrums_all, test_constructed_spectrums], axis=0)
                test_sensor_incident_field_all = np.concatenate(
                    [test_sensor_incident_field_all, test_sensor_incident_field], axis=0)
                test_ground_truth_all = np.concatenate([test_ground_truth_all, np.stack(test_bag[1])], axis=0)
                test_filenames_all = np.concatenate([test_filenames_all, np.stack(test_bag[2])], axis=0)
                test_images_all = np.concatenate([test_images_all, np.stack(test_bag[0])], axis=0)
                #test_lens_all = np.concatenate([test_lens_all, lens], axis=0)
                #test_target_distance_all.append(target_distance)
                print('Test: now the validation loss is: ',test_loss)#target_depth
            test_mat_count += 1
            test_loss_sum = test_loss_sum + test_loss
        test_target_distance_all = np.array(test_target_distance_all)
        test_loss_ave = test_loss_sum / len(test_collection)
        print('test: now the final test loss is: ', test_loss_ave)

        scipy.io.savemat('all_results/mat/test_best_para.mat', {'ground_truths': test_ground_truth_all,
                                             
                                                      'sensor_incident_image':test_sensor_incident_field_all,
                                                      'psfs': psf,
                                                      'intensity_psf':intensity_psf,
                                                      
                                                      #'sensor_incident_image':validation_sensor_incident_image_all,
                                                      #'lens': lens,
                                                      #'obj_distance':target_depth,
                                                      'wave_all':wave_all,
                                                      'incident_wave_all':incident_wave_all,
                                                      'constructed_spectrums': test_constructed_spectrums_all,
                                                      'filenames': test_filenames_all, 'input_images': test_images_all})

    def train(self, epoch_nums, INTERUPT_INDEX):
        self.train_epoch_writer = tf.compat.v1.summary.FileWriter('all_results/train_epoch/', graph=self.sess.graph)
        self.train_batch_writer = tf.compat.v1.summary.FileWriter('all_results/train_batch/', graph=self.sess.graph)
        self.validation_writer = tf.compat.v1.summary.FileWriter('all_results/validation/', graph=self.sess.graph)
        self.tee = Tee('all_results/log.txt')
        validation_summary_input = tf.placeholder(tf.float32, shape=())
        validation_summary = tf.summary.scalar('loss_validation', validation_summary_input)
        batch_size = BATCH_SIZE
        print("now train is running", TRAINING_CROSS_ITEMS)
        
        trainning_data_generating = DataGenerator(True, TRAINING_CROSS_ITEMS, batch_size)
        validation_data_generating = DataGenerator(True, VALIDATION_CROSS_ITEMS, batch_size)
        validation_batch_nums = validation_data_generating.data_nums // batch_size
        batch_nums = trainning_data_generating.data_nums // batch_size
        
        tf.get_default_graph().finalize()
    
        test_datas = load_data(TEST_CROSS_ITEMS)
        test_data = [[], [], []]
        test_count = 0
        test_collection = []
        for r in test_datas:
            test_data[0].append(r.simulate_image)
            test_data[1].append(r.ground_truth)
            test_data[2].append(r.filename)
            test_count += 1
            if test_count >= BATCH_SIZE:
                test_collection.append(test_data) 
                test_count = 0
                test_data = [[], [], []]
        if test_count != 0:
            test_collection.append(test_data)

        validation_loss_best = 0
        training_loss_last = 0
                                
        for i in range(INTERUPT_INDEX + 1, epoch_nums + 1):
            training_loss_sum = 0
            MSE_loss_sum = 0
            MAE_loss_sum = 0
            percep_loss_sum = 0
            for j in range(batch_nums):
                one_batch = trainning_data_generating.get_one_batch()
                edit_images_blur, edit_images_clear = self.sess.run([self.edit_images_blur, self.edit_images_clear], feed_dict={self.raw_images_blur: np.stack(one_batch[0]), self.raw_images_clear: np.stack(one_batch[1])})
                                      
                training_loss, _, training_summary, MSE_loss,MAE_loss, percep_loss,  SSIM_loss = self.sess.run(
                    [
                     self.total_loss,
                     self.train_step_adam,
                     self.scalar_summaries,
                     self.MSE_loss,
                     self.MAE_loss,
                     self.percep_loss,
                     #self.target_depth,
                     self.SSIM_loss
                     ],
                    feed_dict=
                    {self.dropout: DROPOUT,
                     self.learning_rate: LEARNING_RATE,
                     self.reconstructed_images: np.stack(edit_images_blur),
                     self.ground_truth: np.stack(edit_images_clear),
                     self.is_training: True
                     })
                if OPTICAL_MODEL_TRAINING_ENABLE:
                    training_loss, _, training_summary, MSE_loss, MAE_loss, percep_loss= self.sess.run(
                        [
                         self.total_loss,
                         self.train_step_adam_optics,
                         self.scalar_summaries,
                         self.MSE_loss,
                         self.MAE_loss,
                         self.percep_loss,
                         #self.target_depth
                         ],
                        feed_dict=
                        {self.dropout: DROPOUT,
                         self.learning_rate: LEARNING_RATE,
                         self.reconstructed_images: np.stack(edit_images_blur),
                         self.ground_truth: np.stack(edit_images_clear),
                         self.is_training: True
                         })
                # scipy.io.savemat('all_results/' + str(j) + '_tmp.mat', {'image': np.stack(edit_images[0]), 'filename': np.stack(one_batch[2])})
                
                training_loss_sum += training_loss
                MSE_loss_sum +=MSE_loss
                MAE_loss_sum +=MAE_loss
                percep_loss_sum += percep_loss
                # if training_loss > 2 * training_loss_last:
                # scipy.io.savemat('all_results/mat/training_error_' + str(j) + '.mat',
                          #          {'image_blur': np.stack(edit_images_blur[0]),'image_clear': np.stack(edit_images_clear[0]), 'filename': np.stack(one_batch[2])})
                print('the', j, 'in batch_nums ', [training_loss, MSE_loss, MAE_loss, percep_loss, SSIM_loss])
                self.train_batch_writer.add_summary(training_summary, i * batch_nums + j)
                training_loss_last = training_loss

            training_loss_ave = training_loss_sum / batch_nums
            MSE_loss_ave = MSE_loss_sum / batch_nums
            MAE_loss_ave = MAE_loss_sum / batch_nums
            percep_loss_ave = percep_loss_sum/ batch_nums
            self.train_epoch_writer.add_summary(training_summary, i)
            print('training: now the ', i, ' epoch training loss is: ', [training_loss_ave,MSE_loss_ave,MAE_loss_ave, percep_loss_ave])
            # if i == 1 + INTERUPT_INDEX:
            #     training_loss_best = training_loss_ave
            #     ckpt_path = 'all_results/training_ckpt_result/' + str(i) + '.ckpt'
            #     self.saver.save(self.sess, ckpt_path)
            # elif training_loss_best > training_loss_ave:
            #     training_loss_best = training_loss_ave
            #     ckpt_path = 'all_results/training_ckpt_result/' + str(i) + '.ckpt'
            #     self.saver.save(self.sess, ckpt_path)
            print('Now validation begaining -------------------------------------------------')
            validation_loss_sum = 0
            validation_MSE_loss_sum = 0
            validation_MAE_loss_sum = 0
            validation_percep_loss_sum = 0
            validation_mat_count = 0
            for jj in range(validation_batch_nums):
                validation_one_batch = validation_data_generating.get_one_batch()
                edit_images_blur, edit_images_clear = self.sess.run([self.edit_images_blur, self.edit_images_clear],
                                                                    feed_dict={self.raw_images_blur: np.stack(validation_one_batch[0]), self.raw_images_clear: np.stack(validation_one_batch[1])})
                      
                validation_loss, _, validation_MSE_loss, validation_MAE_loss, validation_percep_loss,  SSIM_loss, validation_constructed_spectrums, psf, intensity_psf, incident_wave_all, wave_all, validation_sensor_incident_image = self.sess.run(
                    [
                     self.total_loss,
                     self.train_step_adam,
                     self.MSE_loss,
                     self.MAE_loss,
                     self.percep_loss,
                     #self.target_depth,
                     self.SSIM_loss,
                     self.conv10,
                     self.psf,
                    # self.lens,
                     self.intensity_psf,
                     self.incident_wave_all,
                    # self.focal,
                     self.wave_all,
                     self.sensor_incident_image
                     ],
                    feed_dict=
                    {self.dropout: DROPOUT,
                     self.learning_rate: VALIDATION_LEARNING_RATE,
                     self.reconstructed_images: np.stack(edit_images_blur),
                     self.ground_truth: np.stack(edit_images_clear),
                     self.is_training: True
                     })
                if (i-1)%50 == 0:
                    if validation_mat_count == 0:
                        validation_constructed_spectrums_all = validation_constructed_spectrums
                        validation_sensor_incident_image_all = validation_sensor_incident_image
                        validation_input_images_all = edit_images_blur
                        validation_ground_truth_all = edit_images_clear
                       # validation_sensor_image_all = sensor_image
                       # psf_all = psf
                       # lens_all = lens
                       # complex_wave_all = complex_wave                    
                       # optics_images_all = optics_images
                       # sensor_images_all = sensor_images
                    else:
                        validation_constructed_spectrums_all = np.concatenate(
                            [validation_constructed_spectrums_all, validation_constructed_spectrums], axis=0)
                        validation_sensor_incident_image_all = np.concatenate(
                          [validation_sensor_incident_image_all, validation_sensor_incident_image], axis=0)
                        validation_input_images_all = np.concatenate(
                            [validation_input_images_all, edit_images_blur], axis=0)
                        validation_ground_truth_all = np.concatenate(
                            [validation_ground_truth_all, edit_images_clear], axis=0)
                       # validation_sensor_image_all = np.concatenate(
                       #     [validation_sensor_image_all, sensor_image], axis=0)
                       # psf_all = np.concatenate([psf_all, psf], axis=0)
                       # lens_all = np.concatenate([lens_all, lens], axis=0)
                       # complex_wave_all = np.concatenate([complex_wave_all, complex_wave], axis=0)
                       # optics_images_all = np.concatenate(
                           # [optics_images_all, optics_images], axis=0)
                       # sensor_images_all = np.concatenate([sensor_images_all, sensor_images], axis=0)
                        
                    validation_mat_count += 1
                    
                validation_loss_sum += validation_loss
                validation_MSE_loss_sum +=validation_MSE_loss
                validation_MAE_loss_sum +=validation_MAE_loss
                validation_percep_loss_sum += validation_percep_loss
                # if training_loss > 2 * training_loss_last:
                # scipy.io.savemat('all_results/mat/training_error_' + str(j) + '.mat',
                          #          {'image_blur': np.stack(edit_images_blur[0]),'image_clear': np.stack(edit_images_clear[0]), 'filename': np.stack(one_batch[2])})
                print('the', jj, 'in validation_batch_nums ', [validation_loss, validation_MSE_loss, validation_MAE_loss, validation_percep_loss, SSIM_loss ])#target_depth
                #self.train_batch_writer.add_summary(training_summary, i * batch_nums + j)
                #training_loss_last = training_loss

            validation_loss_ave = validation_loss_sum / validation_batch_nums
            validation_MSE_loss_ave = validation_MSE_loss_sum / validation_batch_nums
            validation_MAE_loss_ave = validation_MAE_loss_sum / validation_batch_nums
            validation_percep_loss_ave = validation_percep_loss_sum/ validation_batch_nums
            self.validation_writer.add_summary(validation_summary.eval(feed_dict={validation_summary_input: validation_loss_ave}), i)
            #self.train_epoch_writer.add_summary(training_summary, i)
            print('Validation: now the ', i, ' epoch validation loss is: ', [validation_loss_ave,validation_MSE_loss_ave, validation_MAE_loss_ave,
                                                                             validation_percep_loss_ave])#target_depth
            if (i-1)%50 == 0:   
                scipy.io.savemat('all_results/mat/validation_' + str(i) + '.mat',
                                        {'constructed_spectrums': validation_constructed_spectrums_all,
                                          # 'optics_images': optics_images_all,
                                          'input_images':validation_input_images_all,
                                          'ground_truths':validation_ground_truth_all,
                                          'intensity_psf':intensity_psf,
                                          'psfs': psf,
                                          'sensor_incident_image':validation_sensor_incident_image_all,
                                          #'lens': lens,
                                          #'obj_distance':target_depth,
                                          'wave_all':wave_all,
                                          'incident_wave_all':incident_wave_all,
                                          'focal_length':focal_length})
                                          #'complex_waves': complex_wave_all})
            # validation_loss_ave = validation_loss_sum / len(validation_collection)
            #print('validation: now the ', i, ' epoch validation loss is: ', validation_loss_ave)
           
            if i == 1 + INTERUPT_INDEX:
                validation_loss_best = validation_loss_ave
                ckpt_path = 'all_results/ckpt_result/' + str(i) + '.ckpt'
                self.saver.save(self.sess, ckpt_path)
            elif validation_loss_best > validation_loss_ave:
                validation_loss_best = validation_loss_ave
                ckpt_path = 'all_results/ckpt_result/' + str(i) + '.ckpt'
                self.saver.save(self.sess, ckpt_path)
            


        test_loss_sum = 0
        test_constructed_spectrums_all = []
        test_sensor_incident_image_all = []
        test_ground_truth_all = []
        test_filenames_all = []
        test_mat_count = 0
        test_images_all = []
        for test_bag in test_collection:
            test_loss, test_constructed_spectrums, test_sensor_incident_image = self.sess.run(
                [self.loss,
                 self.conv10,
                 self.sensor_incident_image
                 ],
                feed_dict=
                {self.dropout: 1.0,
                 self.reconstructed_images: np.stack(test_bag[0]),
                 self.ground_truth: np.stack(test_bag[1]),
                 self.is_training: True
                 })
            if test_mat_count == 0:
                test_constructed_spectrums_all = test_constructed_spectrums
                test_sensor_incident_image_all = test_sensor_incident_image
                test_ground_truth_all = np.stack(test_bag[1])
                test_filenames_all = np.stack(test_bag[2])
                test_images_all = np.stack(test_bag[0])
            else:
                test_constructed_spectrums_all = np.concatenate(
                    [test_constructed_spectrums_all, test_constructed_spectrums], axis=0)
                test_sensor_incident_image_all = np.concatenate(
                    [test_sensor_incident_image_all, test_sensor_incident_image], axis=0)
                test_ground_truth_all = np.concatenate([test_ground_truth_all, np.stack(test_bag[1])], axis=0)
                test_filenames_all = np.concatenate([test_filenames_all, np.stack(test_bag[2])], axis=0)
                test_images_all = np.concatenate([test_images_all, np.stack(test_bag[0])], axis=0)
            test_mat_count += 1
            test_loss_sum = test_loss_sum + test_loss
        test_loss_ave = test_loss_sum / len(test_collection)
        print('test: now the final test loss is: ', test_loss_ave)

        scipy.io.savemat('all_results/mat/test.mat', {'ground_truths': test_ground_truth_all,
                                                      'psfs': psf,
                                                      'sensor_incident_image':test_sensor_incident_image_all,
                                                      'constructed_spectrums': test_constructed_spectrums_all,
                                                      'filenames': test_filenames_all, 'images': test_images_all})
