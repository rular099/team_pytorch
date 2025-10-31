import json
import logging
import numpy as np
from distributed import is_master
from mixdataloaders.loaders import diting2_publish_igp,DiTing330km_publish,PNW,STEAD,MLAAPDE,CREDITX1,INSTANCE,CEA09_20,CSNCD
from mixdataloaders.loaders import SharedEpoch,DataInfo
import os

from loaders import BaseLoaders

def get_dataset_fn(dataset_type):
    if dataset_type == "diting2_publish_igp":
        return diting2_publish_igp
    elif dataset_type == "DiTing330km_publish":
        return DiTing330km_publish
    elif dataset_type == 'PNW':
        return PNW
    elif dataset_type == 'STEAD':
        return STEAD
    elif dataset_type == 'MLAAPDE':
        return MLAAPDE
    elif dataset_type == 'CREDITX1':
        return CREDITX1
    elif dataset_type == 'INSTANCE':
        return INSTANCE
    elif dataset_type == 'CEA09_20':
        return CEA09_20
    elif dataset_type == 'CSNCD':
        return CSNCD

class MixingDataLoader:
    """Mixing different datasets with round-robin or weighted round-robin
    Adpated from https://github.com/mlfoundations/open_clip/pull/107/files
    """
    def __init__(self, args, epoch, sample_weights=False):
        train_index_list = args.train_index_list.split(';')
        train_hdf5_list = args.train_hdf5_list.split(';')
        dataset_type_list = args.dataset_type_list.split(';')
        sample_num_list = [int(element) for element in args.sample_num_list.split(';')] if args.sample_num_list else [-1 for _ in range(len(train_index_list))]
        assert len(train_index_list) == len(train_hdf5_list) == len(dataset_type_list) == len(sample_num_list)
        # if not args.train_num_samples_list or len(args.train_num_samples_list) != len(train_index_list):
        #     train_num_samples_list = [args.train_num_samples//len(train_index_list) for _ in range(len(train_index_list))]
        # else:
        #     train_num_samples_list = args.train_num_samples_list

        data_train = []
        for train_data, hdf5_path, dataset_type,sample_num in zip(train_index_list, train_hdf5_list, dataset_type_list,sample_num_list):
            print(f"{train_data}\t{hdf5_path}\t{dataset_type}")
            args.train_data = os.path.join(args.index_data_folder,train_data)
            args.hdf5_path = os.path.join(args.data_path,hdf5_path)
            assert os.path.exists(args.train_data), f"Dataset file {args.train_data} not found."
            assert os.path.exists(args.hdf5_path), f"HDF5 file {args.hdf5_path} not found."
            args.sample_num = sample_num
            data_train.append(
                get_dataset_fn(dataset_type)(
                    args, is_train=True)
            )

        self.args = args
        self.num_datasets = len(data_train)
        self.dataloaders = [dataset.dataloader for dataset in data_train]
        self.dataiters = [iter(dataloader) for dataloader in self.dataloaders]
        self.datasets = train_index_list
        self.num_batches = sum([dataloader.num_batches for dataloader in self.dataloaders])
        self.num_samples = sum([dataloader.num_samples for dataloader in self.dataloaders])

        # calculate sample weights according to num_samples of multiple datasets
        self.sample_weights = np.array([float(dataloader.num_samples) / self.num_samples for dataloader in self.dataloaders]) if sample_weights else None

        print("Training datasets with virtual epcoh samples in MixingDataLoader:")
        for i, dataset in enumerate(train_index_list):
            print(f"\t{self.dataloaders[i].num_samples} samples per virtual epoch -> {dataset}")
        print(f"Num of datasets in MixingDataLoader: {self.num_datasets}")
        print(f"Num of samples in MixingDataLoader: {self.num_samples}")
        print(f"Num of batches in MixingDataLoader: {self.num_batches}")
        if self.sample_weights is None:
            print("Disable sample_weights...")
        else:
            print(f"Enable sample_weights: {self.sample_weights}")
            for i, dataset in enumerate(train_index_list):
                print(f"\t{self.sample_weights[i]} ratio per virtual epoch -> {dataset}")

        self.count = 0
        self.current_epoch = epoch #0
        self.data_train = data_train
        if self.args.distributed and data_train is not None:
            for data_info in data_train:
                data_info.set_epoch(epoch)
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        while True:
            if self.count == self.num_batches:
                self.current_epoch += 1
                self.count = 0
                if self.args.distributed and self.data_train is not None:
                    for data_info in self.data_train:
                        data_info.set_epoch(self.current_epoch)
                return # end each epoch

            # set random seed for sampling from the same dataset.
            # sample a dataset according to sample_weights
            if self.sample_weights is not None:
                stable_random_seed = int(self.count + self.num_batches * self.current_epoch)
                np.random.seed(stable_random_seed)
                iter_index = np.random.choice(range(self.num_datasets), p=self.sample_weights)
            else:
                iter_index = self.count % self.num_datasets
            # generate training image-text pairs from the sampled dataset.
            try:
                data_iter = self.dataiters[iter_index]
                batch = next(data_iter)
            except StopIteration:
                # refresh dataiter if dataloader is used up.
                self.dataiters[iter_index] = iter(self.dataloaders[iter_index])
                data_iter = self.dataiters[iter_index]
                batch = next(data_iter)

            self.count += 1

            yield batch

def get_mixing_dataset_fn(args, epoch):
    dataloader = MixingDataLoader(args, epoch,args.sample_weights)
    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    return DataInfo(dataloader, shared_epoch)

def get_data(args, epoch=0):
    data = {}
    if args.train_index_list:
        data["train"] = get_mixing_dataset_fn(args, epoch)
    if args.val_index_list:
        data["val"] = get_mixing_dataset_fn(args, epoch)
    return data