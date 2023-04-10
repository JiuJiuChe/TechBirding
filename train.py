'''
@File    :   train.py
@Time    :   2023/04/06 21:30:12
@Author  :   jiujiuche 
@Version :   1.0
@Contact :   jiujiuche@gmail.com
@License :   (C)Copyright 2023-2024, jiujiuche
@Desc    :   main entrance to load the data and train the network
'''

import os
from datetime import datetime
from timeit import default_timer as timer 

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

from data import loader
from tb_utils import base_operation, io_utils, nn_utils

# nn settings
device = nn_utils.get_device()
batch_size = 32
n_workders = 4
input_size = 224
learn_rate = 1e-4
epochs = 60
visualize = False
tag = f'mobilenetv3_{datetime.now().strftime("%m-%d-%Y")}'      # unique fold name to save model
save_epoch = 5                                                  # save model every n epoch
torch.manual_seed(22) 
torch.cuda.manual_seed(22)

# file paths
train_dir = './data/birds515/train'
valid_dir = './data/birds515/valid'
test_dir  = './data/birds515/test'

opt_nn_train = base_operation.BaseOperation('./data/birds515/', 'MobileNetV3_train')
opt_nn_train_handle = opt_nn_train.get_handle(force_run=False, verbose=False)

@opt_nn_train_handle
def train(model, train_loader, valid_loader, optimizer, loss_fn, epochs, device, save_epoch, model_save_dir):
    train_loss_list, train_acc_list = np.zeros(epochs), np.zeros(epochs)
    valid_loss_list, valid_acc_list = np.zeros(epochs), np.zeros(epochs)

    for epoch in tqdm(range(epochs)):
        start_time = timer()
        train_loss, train_acc = nn_utils.train_step(
            model=model, dataloader=train_loader, loss_fn=loss_fn, optimizer=optimizer, \
            device=device)
        valid_loss, valid_acc = nn_utils.test_step(
            model=model, dataloader=valid_loader, loss_fn=loss_fn, device=device)
        end_time = timer()

        print(
            f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"valid_loss: {valid_loss:.4f} | valid_acc: {valid_acc:.4f} | "
            f"time: {end_time-start_time:.3f}s"
        )

        train_loss_list[epoch], train_acc_list[epoch] = train_loss, train_acc
        valid_loss_list[epoch], valid_acc_list[epoch] = valid_loss, valid_acc

        if epoch % save_epoch == 0 and epoch != 0:
            torch.save(network.state_dict(), os.path.join(model_save_dir, f'model_{epoch}.pth'))

    return train_loss_list, train_acc_list, valid_loss_list, valid_loss_list

if __name__ == '__main__':
    # make model save dir
    model_save_dir = f'./models/birds515/{tag}'
    if (not os.path.exists(model_save_dir)):
        os.makedirs(model_save_dir)
    # get dataset stats
    train_list = io_utils.dfs_files(train_dir, '.jpg')
    valid_list = io_utils.dfs_files(valid_dir, '.jpg')
    opt_ds_stats = base_operation.BaseOperation('./data/birds515/', 'train_valid_stats')
    opt_ds_stats_handle = opt_ds_stats.get_handle(force_run=False, verbose=False)
    mean, std = opt_ds_stats.run_opt(loader.running_stats, force_run=False, verbose=False, 
                                    images=train_list+valid_list)

    # transforms for image
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(int(input_size/0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # define the dataset
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(root=valid_dir, transform=valid_transform)
    test_data  = datasets.ImageFolder(root=test_dir,  transform=valid_transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, \
                                num_workers=n_workders)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, \
                                num_workers=n_workders)
    test_dataloader =  DataLoader(test_data, batch_size=batch_size, shuffle=False, \
                                num_workers=n_workders)

    # train network
    network = models.mobilenet_v3_small(pretrained=True).to(device)
    if visualize:
        summary(network, input_size=[1, 3, input_size, input_size])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=network.parameters(), lr=learn_rate)
    train_loss, train_acc, valid_loss, valid_acc = train(
        network, train_dataloader, valid_dataloader, optimizer, loss_fn, epochs, device, \
        save_epoch, model_save_dir)

    torch.save(network.state_dict(), os.path.join(model_save_dir, 'model_final.pth'))

    # plot training curves 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(epochs)+1, train_loss, label='train loss')
    plt.plot(np.arange(epochs)+1, valid_loss, label='valid loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(epochs)+1, train_acc, label='train accuracy')
    plt.plot(np.arange(epochs)+1, valid_acc, label='valid accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
