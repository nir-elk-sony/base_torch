# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 07:50:25 2024

@author: 7000028246
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:21:39 2024

@author: 7000028246
"""


from torchvision import datasets
import torch
# from torchvision.transforms import transforms

from torchvision import transforms
import os
import random
from PIL import Image

# Define transformations for the validation set
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def validation_preprocess():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def random_crop_flip_preprocess():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])



# Define representative dataset generator
def get_representative_dataset(src_dir, val_transform = val_transform, reshuffle_dir = True, reshuffle_gen = True, n_iter = None, wrap_res_list = False, do_unsqueeze = False):

    image_file_list = os.listdir(src_dir)
    if reshuffle_dir:
        random.shuffle(image_file_list)


    def representative_dataset():
        file_order = image_file_list.copy()
        if reshuffle_gen:
            random.shuffle(file_order)
        img_cnt = 0
        for fname in file_order:
            image = Image.open(f'{src_dir}/{fname}').convert('RGB')
            image = val_transform(image)
            
            r = image.clone()
            if do_unsqueeze:
                r = r.unsqueeze(0)
            
            if wrap_res_list:
                yield [r]
            else:
                yield r
            img_cnt += 1
            if n_iter == img_cnt:
                return
            
    return representative_dataset



def get_init_rep_dset(train_dir):
    init_dataset = datasets.ImageFolder(train_dir, random_crop_flip_preprocess())

    return init_dataset


def get_representative_dataset_imagenet(train_dir, shuffle, batch_size=1):
    init_dataset = get_init_rep_dset(train_dir)
    representative_data_loader = torch.utils.data.DataLoader(init_dataset,
                                                             batch_size=batch_size,
                                                             shuffle=shuffle,
                                                             num_workers=4,
                                                             pin_memory=True)
    return representative_data_loader


def get_train_samples(train_loader, num_samples):
    train_data = []
    labels = []
    for batch in train_loader:
        train_data.append(batch[0])
        labels.append(batch[1])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    random_samples_indices = torch.randperm(num_samples)[:num_samples]
    return torch.cat(train_data, dim=0)[random_samples_indices], torch.cat(labels, dim=0)[random_samples_indices]
