import os
import json
import numpy as np
from PIL import Image
import torch
import copy
from torch.utils.data import Dataset, default_collate
from torchvision import transforms

def wrap_dataset(dataset, split, args):
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=(split == 'train'),
        num_workers=args.workers, 
        pin_memory=True, 
        collate_fn=default_collate,
    )
    return loader

def get_metadata(dataset_name):
    if dataset_name == 'COCO2014':
        meta = {
            'num_classes': 80,
            'path_to_dataset': 'data/coco2014',
            'path_to_images': 'data/coco2014'
        }
    elif dataset_name == 'ML48S':
        meta = {
            'num_classes': 100,
            'path_to_dataset': 'data/ml48s',
            'path_to_images': 'data/ml48s'
        }
    else:
        raise NotImplementedError('Metadata dictionary not implemented.')
    return meta

def get_imagenet_stats():
    '''
    Returns standard ImageNet statistics. 
    '''
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    return (imagenet_mean, imagenet_std)

def get_transforms(flip=True):
    '''
    Returns image transforms.
    '''
    
    (imagenet_mean, imagenet_std) = get_imagenet_stats()
    tx = {}
    if flip:
        tx['train'] = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        tx['train'] = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    tx['val'] = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['test'] = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return tx

def generate_split(num_ex, frac, rng):
    '''
    Computes indices for a randomized split of num_ex objects into two parts,
    so we return two index vectors: idx_1 and idx_2. Note that idx_1 has length
    (1.0 - frac)*num_ex and idx_2 has length frac*num_ex. Sorted index sets are 
    returned because this function is for splitting, not shuffling. 
    '''
    
    # compute size of each split:
    n_2 = int(np.round(frac * num_ex))
    n_1 = num_ex - n_2
    
    # assign indices to splits:
    idx_rand = rng.permutation(num_ex)
    idx_1 = np.sort(idx_rand[:n_1])
    idx_2 = np.sort(idx_rand[-n_2:])
    
    return (idx_1, idx_2)

def get_dataset(args):
    '''
    Given a parameter dictionary P, initialize and return the specified dataset. 
    '''
    
    # define transforms:
    tx = get_transforms(flip=('ML48S' not in args.dataset))
    # select and return the right dataset:
    ds = multilabel(args, tx).get_datasets()
    return ds

def load_data(base_path, mask, phases):
    data = {}
    for phase in phases:
        data[phase] = {}
        data[phase]['labels'] = np.load(os.path.join(base_path, '{}_labels.npy'.format(phase)))
        data[phase]['mask'] = np.load(os.path.join(base_path, '{}_{}_mask.npy'.format(mask[phase], phase)))
        data[phase]['images'] = np.load(os.path.join(base_path, '{}_images.npy'.format(phase)))
    return data

class multilabel:

    def __init__(self, args, tx):
        
        # get dataset metadata:
        meta = get_metadata(args.dataset)
        self.base_path = meta['path_to_dataset']
        
        # load data:
        phases = ['train', 'val']
        if 'ML48S' in args.dataset:
            phases.append('test')
        source_data = load_data(self.base_path, mask={'train':args.train_set_variant, 'val': args.val_set_variant, 'test': args.val_set_variant}, phases=phases)
        full_data = load_data(self.base_path, mask={'train':'full', 'val': 'full', 'test': 'full'}, phases=phases)
        
        if 'test' not in source_data:
            # generate indices to split official train set into train and val:
            split_idx = {}
            (split_idx['train'], split_idx['val']) = generate_split(
                len(source_data['train']['images']),
                args.val_frac,
                np.random.RandomState(args.split_seed)
            )

            for phase in ['train', 'val']:
                if os.path.exists(os.path.join(self.base_path, 'seed_{}_{}_split.npy'.format(args.split_seed, phase))):
                    if args.val_frac == 0.2:
                        assert np.all(split_idx[phase] == np.load(os.path.join(self.base_path, 'seed_{}_{}_split.npy'.format(args.split_seed, phase))))
                else:
                    np.save(os.path.join(self.base_path, 'seed_{}_{}_split.npy'.format(args.split_seed, phase)), split_idx[phase])

            for data in [source_data, full_data]:
                data['test'] = data['val']
                for split in ['val', 'train']:
                    data[split]['images'] = data['train']['images'][split_idx[split]]
                    data[split]['labels'] = data['train']['labels'][split_idx[split]]
                    data[split]['mask'] = data['train']['mask'][split_idx[split]]
        
        # define train set:
        self.train = ds_multilabel(
            args.dataset,
            source_data['train']['images'],
            source_data['train']['labels'],
            source_data['train']['mask'],
            tx['train'],
            train=True,
            remove_unannotated=args.remove_unannotated, remove_empty = args.remove_empty
        )

        # define val set:
        self.val = ds_multilabel(
            args.dataset,
            full_data['val']['images'],
            full_data['val']['labels'],
            full_data['val']['mask'],
            tx['val']
        )
        
        # define test set:
        self.test = ds_multilabel(
            args.dataset,
            full_data['test']['images'],
            full_data['test']['labels'],
            full_data['test']['mask'],
            tx['test']
        )
        
        # define dict of dataset lengths: 
        self.lengths = {'train': len(self.train), 'val': len(self.val), 'test': len(self.test)}
    
    def get_datasets(self):
        return {'train': self.train, 'val': self.val, 'test': self.test}

class ds_multilabel(Dataset):

    def __init__(self, dataset_name, image_ids, label_matrix, mask, tx, train=False, remove_empty=False, remove_unannotated=False):
        meta = get_metadata(dataset_name)
        self.num_classes = meta['num_classes']
        self.path_to_images = meta['path_to_images']
        
        self.image_ids = image_ids
        self.label_matrix = label_matrix
        self.mask = mask
        self.tx = tx

        if train:
            if remove_empty:
                empty = np.all(np.logical_not(np.logical_and(self.label_matrix, self.mask)), axis=1)
                self.image_ids = self.image_ids[np.logical_not(empty)]
                self.label_matrix = self.label_matrix[np.logical_not(empty)]
                self.mask = self.mask[np.logical_not(empty)]
            if remove_unannotated:
                annotated = np.logical_not(np.all(self.mask == 0, axis=1))
                self.image_ids = self.image_ids[annotated]
                self.label_matrix = self.label_matrix[annotated]
                self.mask = self.mask[annotated]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,idx):
        image_path = os.path.join(self.path_to_images, self.image_ids[idx])
        with Image.open(image_path) as I_raw:
            I = self.tx(I_raw.convert('RGB'))
        
        out = {
            'image': I,
            'labels': torch.FloatTensor(np.copy(self.label_matrix[idx, :])),
            'mask': torch.FloatTensor(np.copy(self.mask[idx, :])),
            'idx': idx,
            'path': self.image_ids[idx]
        }
        
        return out
