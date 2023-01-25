'''
Name: Sumegha Singhania, Erica Shephard, Nicholas Wen
Class: CS7180 Advanced Perception
Final Project

This file contains functions to prepare the training
and testing dataset for the network.
'''
import pydicom as dicom
import numpy as np
import cv2
import pandas as pd
import shutil
import random
import imutils
import torch
from os import listdir
from os.path import join

# check if file is an image
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
# check if file is a dcm file
def is_dicom_file(filename):
    return any(filename.endswith(extension) for extension in ['.dcm', '.DCM'])

# function to convert dicom images to jpg
def conv_dcm_jpg(filename):
    ds = dicom.dcmread(filename)
    pixel_array_numpy = ds.pixel_array
    img = cv2.merge((pixel_array_numpy,pixel_array_numpy,pixel_array_numpy))
    image_format = '.jpg'
    filename = filename.replace('.dcm', image_format)
    cv2.imwrite(filename, img)

# erase pneumonia pixels and add to training dataset
def erase_pneumonia(label_path,data_path):
    label_df = pd.read_csv(label_path)
    # filtering out pneumonia samples
    label_df = label_df[label_df['Target']==1]
    for ind in label_df.index:
        # Extracting bounding box information
        id = label_df['patientId'][ind]
        x = label_df['x'][ind].astype(int)
        y = label_df['y'][ind].astype(int)
        w = label_df['width'][ind].astype(int)
        h = label_df['height'][ind].astype(int)

        image_path = f"{data_path}{id}.jpg"
        erased_path = f"{data_path}{id}_erased.jpg"

        # replacing pneumonia bb with average value
        shutil.copy2(image_path,erased_path) 
        erased = cv2.imread(erased_path,0)
        avg = np.mean(erased)
        # print (avg)
        # cv2.imshow('pneumonia_image', img)
        
        # defining bounding box upper bounds
        x_upper_bound=x+w
        y_upper_bound=y+h
        if x+w>erased.shape[1]:
            x_upper_bound = erased.shape[1]
        if y+h>erased.shape[0]:
            y_upper_bound = erased.shape[0]

        # replacing bounding box with average image pixels
        for i in range(x,x_upper_bound):
            for j in range(y,y_upper_bound):
                erased[i,j] = avg
        
        # cv2.imshow('erased_image', erased)

        erased_img = cv2.merge((erased,erased,erased))
        cv2.imwrite(erased_path, erased_img)

# scale down the images by specified factor
def resize(filename,scale_factor):
    img = cv2.imread(filename,0)
    resized = cv2.resize(img, (0, 0), fx = scale_factor, fy = scale_factor)
    resized_img = cv2.merge((resized,resized,resized))
    cv2.imwrite(filename,resized_img)

# applying data augmentation to expand the dataset
def data_augmentation(filename,flag):
    img = cv2.imread(filename,0)
    aug = img.copy()

     # rotate
    if flag == 0:
        angle = random.randrange(10,350)
        aug = imutils.rotate_bound(aug, angle)
        aug = cv2.resize(aug,img.shape)
    else:
        # flip
        axis = random.randint(0,1)
        aug = cv2.flip(aug, axis)
        aug = cv2.resize(aug,img.shape)
    
    filename = filename[:-4]
    filename = f"{filename}_augmented.jpg"
    cv2.imwrite(filename,aug)

     
# preparing the training dataset
def training_data(data_path,label_path,scale_factor):
    image_filenames = [join(data_path,x) for x in listdir(data_path) if is_dicom_file(x)]
       
    read dicom images and convert them to jpeg 
    for x in image_filenames:
        conv_dcm_jpg(x)
    
    # replace penumonia pixels with average image value
    erase_pneumonia(label_path,data_path)

    # # resize all jpg images and apply data augmentation
    jpeg_image_filenames = [join(data_path,x) for x in listdir(data_path) if is_image_file(x)]
    for x in jpeg_image_filenames:
        resize(x,scale_factor)
        flag = random.randint(0,1)
        data_augmentation(x,flag)

    # return final training dataset
    train_dataset = [join(data_path,x) for x in listdir(data_path) if is_image_file(x)]

    return train_dataset

def testing_data(data_path,scale_factor):
    image_filenames = [join(data_path,x) for x in listdir(data_path) if is_dicom_file(x)]

    # read dicom images and convert them to jpeg 
    for x in image_filenames:
        conv_dcm_jpg(x)

    # resize all jpg images
    jpeg_image_filenames = [join(data_path,x) for x in listdir(data_path) if is_image_file(x)]
    for x in jpeg_image_filenames:
        resize(x,scale_factor)

    return jpeg_image_filenames

