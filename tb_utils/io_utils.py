'''
@File    :   io_utils.py
@Time    :   2023/04/05 22:01:50
@Author  :   jiujiuche 
@Version :   1.0
@Contact :   jiujiuche@gmail.com
@License :   (C)Copyright 2023-2024, jiujiuche
@Desc    :   File, I/O related functions
'''
import os
import pickle
import numpy as np
from PIL import Image

def create_dir(dir_path):
    """Create directory if it does not exists

    Args:
        dir_path (str): destination folder
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def omni_save(file_name, data):
    """auto save data given the file extension

    Args:
        file_name (str): path to file saving destination
        data (str): corresponding data to be saved (e.g., string for .txt files) 

    Raises:
        NotImplementedError: not supported extension type
    """
    _, ext = os.path.splitext(file_name)
    if ext == '.npy':
        np.save(file_name, data)
    elif ext == '.txt':
        with open(file_name, 'w', encoding='utf8') as f:
            f.writelines(data)
    elif ext == '.pkl' or ext == '.pickle':
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
    elif ext == '.jpg' or ext == '.png':
        Image.fromarray(data).save(file_name)
    else:
        raise NotImplementedError(f'Format {ext} is not supported')

def omni_load(file_name):
    """auto load data given the file extension

    Args:
        file_name (str): path to load the file

    Raises:
        NotImplementedError: not supported extension type

    Returns:
        data from the file
    """
    _, ext = os.path.splitext(file_name)
    if ext == '.npy':
        data = np.load(file_name)
    elif ext == '.txt':
        with open(file_name, 'r', encoding='utf8') as f:
            data = f.readlines()
    elif ext == '.pkl' or ext == '.pickle':
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
    elif ext == '.jpg' or ext == '.png':
        data = np.array(Image.open(file_name))
    else:
        raise NotImplementedError(f'Format {ext} is not supported')
    return data

def dfs_files(parent_dir, ext):
    """DFS search of parent folder for all files with given extension

    Args:
        parent_dir (str): root folder path
        ext (str): extension (e.g., jpg)

    Returns:
        list: list of path to files
    """
    file_list = []
    for root, dirs, files in os.walk(parent_dir):
        path = root.split(os.sep)
        for file in files:
            if ext in file:
                file_list.append(os.path.join(os.sep.join(path), file))
    return file_list
