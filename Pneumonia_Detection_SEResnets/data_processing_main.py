'''
Name: Sumegha Singhania, Erica Shephard, Nicholas Wen
Class: CS7180 Advanced Perception
Final Project

This file calls functions to alter the training and testing data.
'''

from data_preprocess_new import training_data,testing_data
import torch.utils.data
from torch.utils.data import DataLoader


data_path = "/Users/sumeghasinghania/Desktop/Projects/Final/stage_2_train_images/"
label_path = "/Users/sumeghasinghania/Desktop/Projects/Final/stage_2_train_labels.csv"
train_path = "/Users/sumeghasinghania/Desktop/Projects/Final/stage_2_test_images"

# Pre-process training data
train_dataset = training_data(data_path,label_path,0.5)
train_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=64, shuffle=True)

# Pre-process testing data
test_dataset = testing_data(train_path,0.5)
test_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=64, shuffle=True)

# drop duplicates from label csv file
path = "/Users/sumeghasinghania/Desktop/CS7180_FinalProject/train_labels.csv"
label_df = pd.read_csv(path)
label_df.drop_duplicates(subset=['patientId', 'Target'], keep='last',inplace=True)

# adding images with _erased in the label database as non-pneumonia scans
temp_df = label_df[label_df['Target']==1]
temp_df["patientId"] = temp_df["patientId"]+"_erased"
temp_df["Target"] = 0
erased_df = label_df.append(temp_df,ignore_index=True)
erased_df.to_csv("/Users/sumeghasinghania/Desktop/CS7180_FinalProject/train_labels_erased.csv",index=False)




