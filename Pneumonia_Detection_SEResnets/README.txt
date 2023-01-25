Sumegha Singhania, Nicholas Wen, Erica Shepherd
12/15/2022
CS 7180 Advanced Perception
Final Project: Pneumonia Detection using SEResNets

OS: Windows 11, MacOS Ventura

This Project was compiled on Google colab on the following notebook:
https://colab.research.google.com/drive/10fpvwOI0knQaXs-vUnRfk9n8PbmrCpfs?usp=sharing

Instructions for terminal:
    - Requires pytorch, torch, torchvision, pandas, pydicom, numpy, and PIL to be installed
    - RSNA dataset downloaded from https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data 
    - Compiled with the following commands in the terminal:
      	$ python data_processing_main.py
        $ python database.py
        $ python network.py [1-5] [num]

The network.py takes in 2 arguments, the first being the network to run with the following model numbers:

Model numbers: 
[1] SEResNet-18 
[2] SEResNet-34 
[3] SEResNet-50 
[4] SEResNet-101
[5] Runs all models

The second argument is the number of epochs to train

Files:
1. data_preprocess.py and data_processing_main.py. - deal with all the pre-processing and preparing data and data labels.
2. mini_dataset_128 - contains the smaller dataset with 128x128 training and testing images.
3. mini_test_labels, mini_train_labels - files containing bounding box and ground truth information for the smaller dataset.
4. test_labels, train_labels, train_labels_erased - files containing bounding box and ground truth information for the larger dataset.
5. network.py - Contains the network models for training and testing.
6. database.py - Contains dataset class used in network and other commands for converting data or preparing data directory.
