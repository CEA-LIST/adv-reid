# @copyright CEA-LIST/DIASI/SIALV/LVA (2020)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import numpy as np
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
from tqdm import tqdm
import os
import pickle

class RandomIdentitySampler(Sampler):
    """
    Sampler creating the minibatch for training so that each minibatch consist
    of batch_size // num_instances identities and each identities has
    num_instances instances.
    
    Arguments:
        data_source (Dataset instance): dataset from which sample the data
        num_instance (int, default: 1): number of images per identity
        save_dict (bool, default: True): saving the dictionary {class_id: index}
            of the dataset on disk
        load_dict (bool, default: True): loading the dictionary of the dataset if it exists
        dataset (str, default: 'market'): name of the dataset for the dictionary filename 
    """
    def __init__(self, data_source, num_instances=1, save_dict=True, load_dict=True, dataset='market'):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        dict_filename = dataset + "_ids.pkl"
        if load_dict and os.path.isfile(dict_filename):
            with open(dict_filename, 'rb') as pickle_file:
                self.index_dic = pickle.load(pickle_file)
            print("Dictionary of ids loaded")
        else:
            for index, sample in tqdm(enumerate(data_source), total=len(data_source), desc="Initializing the batch sampler"):
                self.index_dic[int(sample['id'])].append(index)
            if save_dict:
                with open(dict_filename, 'wb') as pickle_file:
                    pickle.dump(self.index_dic, pickle_file)
                print("Dictionary of ids saved")
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class MarketDuke(Dataset):
    """Market1501 dataset or Duke dataset
    
    Arguments:
        folder (str): folder that contains the Market1501 or Duke dataset
        name (str, default: 'market'): which dataset to process, either 'market' or 'duke'
        set (str, default: 'train'): which split to process, either 'train', 'gallery' or 'probe'
        transform (callable, default: None):  a function/transform that  takes in an PIL image
            and returns a transformed version. E.g, 'transforms.RandomCrop'"""

    def __init__(self, folder, name='market', set='train',
                 transform=None):
        self.folder = folder
        
        if name == 'market':
            self.train_dir = os.path.join(self.folder, 'bounding_box_train')
            self.gallery_dir = os.path.join(self.folder, 'bounding_box_test')
            self.probe_dir = os.path.join(self.folder, 'query')

        elif name == 'duke':
            self.train_dir = os.path.join(self.folder, 'bounding_box_train')
            self.gallery_dir = os.path.join(self.folder, 'bounding_box_test')
            self.probe_dir = os.path.join(self.folder, 'query')
        else:
            print("No dataset with this name")

        self.set = set
        self.transform = transform

        if set == 'train':
            self.files = sorted(glob.glob(os.path.join(self.train_dir,'*.jpg')))
        elif set == 'gallery':
            self.files = sorted(glob.glob(os.path.join(self.gallery_dir,'*.jpg')))
            self.files = [f for f in self.files if not ("-1_" in f)]
        elif set == 'probe':
            self.files = sorted(glob.glob(os.path.join(self.probe_dir,'*.jpg')))
        else:
            print('No set with this name')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        image = Image.open(self.files[idx])

        if self.set == 'train':
            id_list = np.unique([np.int(f.split('/')[-1][0:4]) for f in self.files])
            id, = np.where(id_list == np.int(self.files[idx].split('/')[-1][0:4]))
        else:
            id = np.int(self.files[idx].split('/')[-1][0:4])

        cam = np.int(self.files[idx].split('_c')[1][0])

        sample = {'image': image, 'id': np.array(id), 'cam': cam}

        if self.transform:
            sample = {'image': self.transform(image), 'id': np.array(id), 'cam': cam}

        return sample
