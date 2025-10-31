from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from multiprocessing import Value
from dataclasses import dataclass

import logging
import json
import h5py
import h5pickle
import pandas as pdsbatch_out
import numpy as np
import torch
import random

from scipy.signal import resample
try:
    from scipy.signal import tukey
except:
    from scipy.signal.windows import tukey
import os

def is_valid(p,s,LSD_hdf5_sample):
    if LSD_hdf5_sample is None:
        return False
    if not np.isnan(p) and (p > LSD_hdf5_sample.shape[0] or p < 0):
        return False
    if not np.isnan(s) and (s > LSD_hdf5_sample.shape[0] or s < 0):
        return False
    if np.isnan(p) or np.isnan(s):
        return True
    return p < s
    
class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value
    
@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

    
class JsonDataset(Dataset):
    def __init__(self, json_path, hdf5_path,transforms):
        # logging.debug(f'Loading json data from {json_path}.')
        # logging.debug(f'Loading h5py data from {hdf5_path}.')
        self.keys = list(json.load(open(json_path, 'r')).keys())
        self.HDF_file = h5py.File(hdf5_path, 'r')
        self.transforms = transforms
        # logging.debug('Done loading data.')


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # TODO transform
        sample_key = self.keys[idx]
        t_data = self.HDF_file.get('earthquake').get(sample_key)[()]
        t_data = self.transforms(t_data)

        return t_data
    
class Diting2DATASET(Dataset):
    def __init__(self, json_path, hdf5_path,augmentations,sample_num):
        # logging.debug(f'Loading json data from {json_path}.')
        # logging.debug(f'Loading h5py data from {hdf5_path}.')
        assert json_path.endswith('.json'), f"diting2_publish_igp only support json file, but got {json_path}"
        self.json_path = json_path # TODO
        self.json_file = json.load(open(json_path, 'r'))
        self.keys = list(self.json_file.keys())
        if sample_num > 0:
            self.keys = random.sample(self.keys, sample_num)
        print("Diting2DATASET:",len(self.keys))
        self.HDF_file = h5py.File(hdf5_path, 'r')
        self.augmentations = augmentations
        # logging.debug('Done loading data.')


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        sample_key = self.keys[idx]

        columns = ['P_index', 'Pn_index', 'Pg_index']
        try:
            p = next((self.json_file[sample_key][column] for column in columns if column in self.json_file[sample_key].keys()))
        except StopIteration:
            p = np.nan
        columns = ['S_index','Sn_index', 'Sg_index']
        try:
            s = next((self.json_file[sample_key][column] for column in columns if column in self.json_file[sample_key].keys()))
        except StopIteration:
            s = np.nan
            
        if 'non_natural' in self.json_path:
            LSD_hdf5_sample = self.HDF_file.get('non_natural').get(sample_key)[()]
        else:
            LSD_hdf5_sample = self.HDF_file.get('earthquake').get(sample_key)[()]
        
        times = 0
        while not is_valid(p,s,LSD_hdf5_sample):
            idx = random.randint(0, len(self.keys) - 1)
            sample_key = self.keys[idx]

            columns = ['P_index', 'Pn_index', 'Pg_index']
            try:
                p = next((self.json_file[sample_key][column] for column in columns if column in self.json_file[sample_key].keys()))
            except StopIteration:
                p = np.nan
            columns = ['S_index','Sn_index', 'Sg_index']
            try:
                s = next((self.json_file[sample_key][column] for column in columns if column in self.json_file[sample_key].keys()))
            except StopIteration:
                s = np.nan
                
            if 'non_natural' in self.json_path:
                LSD_hdf5_sample = self.HDF_file.get('non_natural').get(sample_key)[()]
            else:
                LSD_hdf5_sample = self.HDF_file.get('earthquake').get(sample_key)[()]
                
            times += 1
            # print(f"{self.json_path} resample times: {times} for data invalid!")
            
        Z_wave = torch.from_numpy(LSD_hdf5_sample[:, 0]).reshape(1, -1).to(dtype=torch.float32) # ->[1,12000]
        N_wave = torch.from_numpy(LSD_hdf5_sample[:, 1]).reshape(1, -1).to(dtype=torch.float32)
        E_wave = torch.from_numpy(LSD_hdf5_sample[:, 2]).reshape(1, -1).to(dtype=torch.float32)

        # length = Z_wave.shape[1]

        wave_list = [Z_wave, N_wave, E_wave]

        integrated_wave = torch.cat(wave_list, dim=0)

        
        # p = LSD_csv_sample['P_index'].values[0] if not pd.isnull(LSD_csv_sample['P_index'].values[0]) else np.nan
        # s = LSD_csv_sample['S_index'].values[0] if not pd.isnull(LSD_csv_sample['S_index'].values[0]) else np.nan

        sample = {'data': integrated_wave,
                  'p': p,
                  's': s,
                #   'data_path':self.json_path,
                # TODO:for debug
                #   "original_data":integrated_wave,
                #   "original_p":p,
                # "original_s":s
                  }
        if self.augmentations is not None:
            x = self.augmentations(sample)  # x['data']:[x_q, x_k] or [x, label]
        else:
            raise Exception()

        return x

class Diting1DATASET(Dataset):
    def __init__(self, csv_path, hdf5_folder_path,augmentations,sample_num):
        # logging.debug(f'Loading csv data from {csv_path}.')
        # logging.debug(f'Loading h5py folder from {hdf5_folder_path}.')
        target_columns = ['Key','P_index', 'Pn_index', 'Pg_index', 'S_index', 'Sn_index', 'Sg_index','part']
        self.csv = pd.read_csv(csv_path, dtype={'Key': str})[target_columns]
        if sample_num > 0:
            self.csv = self.csv.sample(n=sample_num, random_state=42)
        self.csv_path = csv_path
        print(f"Diting1DATASET:{len(self.csv)}")
        self.hdf5_folder_path = hdf5_folder_path
        self.augmentations = augmentations
        self.hdf5_files = []
        for part in range(28):
            self.hdf5_files.append(h5py.File(self.hdf5_folder_path + 'DiTing330km_part_{}.hdf5'.format(part), 'r'))
        # logging.debug('Done loading data.')


    def __len__(self):
        return len(self.csv)
    
    def get_data(self,key,part):
        dataset = self.hdf5_files[part].get('earthquake/'+str(key))    
        return np.array(dataset).astype(np.float32)

    def __getitem__(self, idx):
        LSD_csv_sample = self.csv.iloc[[idx]]
        columns = ['P_index', 'Pn_index', 'Pg_index']
        p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        columns = ['S_index','Sn_index', 'Sg_index']
        s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        
        
        part = int(LSD_csv_sample['part'].iloc[0])
        key_correct = LSD_csv_sample['Key'].iloc[0].split('.')
        key = key_correct[0].rjust(6,'0') + '.' + key_correct[1].ljust(4,'0')
        # LSD_hdf5_sample = data.get_from_DiTing_100Hz(part=part, key=key, h5file_path=self.hdf5_folder_path)
        LSD_hdf5_sample = self.get_data(key,part)
        
        times = 0
        while not is_valid(p,s,LSD_hdf5_sample):
            idx = random.randint(0, len(self.csv) - 1)
            LSD_csv_sample = self.csv.iloc[[idx]]
            columns = ['P_index', 'Pn_index', 'Pg_index']
            p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            columns = ['S_index','Sn_index', 'Sg_index']
            s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            
            part = int(LSD_csv_sample['part'].iloc[0])
            key_correct = LSD_csv_sample['Key'].iloc[0].split('.')
            key = key_correct[0].rjust(6,'0') + '.' + key_correct[1].ljust(4,'0')
            # LSD_hdf5_sample = data.get_from_DiTing_100Hz(part=part, key=key, h5file_path=self.hdf5_folder_path)
            LSD_hdf5_sample = self.get_data(key,part)
            
            times += 1
            # print(f"{self.csv_path} resample times: {times} for data invalid!")
            

        Z_wave = torch.from_numpy(LSD_hdf5_sample[:, 0]).reshape(1, -1).to(dtype=torch.float32) # ->[1,12000]
        N_wave = torch.from_numpy(LSD_hdf5_sample[:, 1]).reshape(1, -1).to(dtype=torch.float32)
        E_wave = torch.from_numpy(LSD_hdf5_sample[:, 2]).reshape(1, -1).to(dtype=torch.float32)

        # length = Z_wave.shape[1]

        wave_list = [Z_wave, N_wave, E_wave]

        integrated_wave = torch.cat(wave_list, dim=0)

        sample = {'data': integrated_wave,
                  'p': p,
                  's': s,
                   # 'data_path':self.csv_path,
                    # TODO:for debug
                #   "original_data":integrated_wave,
                #   "original_p":p,
                # "original_s":s,
                  }
        if self.augmentations is not None:
            x = self.augmentations(sample)  # x['data']:[x_q, x_k] or [x, label]
        else:
            raise Exception()

        return x
    
class PNWDATASET(Dataset):
    def __init__(self, csv_path, hdf5_path,augmentations,sample_num):
        # logging.debug(f'Loading csv data from {csv_path}.')
        # logging.debug(f'Loading h5py from {hdf5_path}.')
        target_columns = ['Key','P_index', 'Pn_index', 'Pg_index', 'S_index', 'Sn_index', 'Sg_index']
        self.csv = pd.read_csv(csv_path, dtype={'Key': str})[target_columns]
        if sample_num > 0:
            self.csv = self.csv.sample(n=sample_num, random_state=42)
        self.csv_path = csv_path
        print(f"PNWDATASET:{len(self.csv)}")
        self.HDF_file = h5py.File(hdf5_path, 'r')
        self.augmentations = augmentations
        # logging.debug('Done loading data.')


    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        LSD_csv_sample = self.csv.iloc[[idx]]
        columns = ['P_index', 'Pn_index', 'Pg_index']
        p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        columns = ['S_index','Sn_index', 'Sg_index']
        s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)

        bucket, array = LSD_csv_sample['Key'].iloc[0].split('$')
        x, y, z = iter([int(i) for i in array.split(',:')])
        LSD_hdf5_sample = np.array(self.HDF_file['data'][bucket][x, :y, :z].transpose()[:, ::-1])
        
        times = 0
        while not is_valid(p,s,LSD_hdf5_sample):
            idx = random.randint(0, len(self.csv) - 1)
            LSD_csv_sample = self.csv.iloc[[idx]]
            columns = ['P_index', 'Pn_index', 'Pg_index']
            p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            columns = ['S_index','Sn_index', 'Sg_index']
            s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            
            bucket, array = LSD_csv_sample['Key'].iloc[0].split('$')
            x, y, z = iter([int(i) for i in array.split(',:')])
            LSD_hdf5_sample = np.array(self.HDF_file['data'][bucket][x, :y, :z].transpose()[:, ::-1])
            
            times += 1
            # print(f"{self.csv_path} resample times: {times} for data invalid!")

        
        Z_wave = torch.from_numpy(LSD_hdf5_sample[:, 0]).reshape(1, -1).to(dtype=torch.float32) # ->[1,12000]
        N_wave = torch.from_numpy(LSD_hdf5_sample[:, 1]).reshape(1, -1).to(dtype=torch.float32)
        E_wave = torch.from_numpy(LSD_hdf5_sample[:, 2]).reshape(1, -1).to(dtype=torch.float32)

        # length = Z_wave.shape[1]

        wave_list = [Z_wave, N_wave, E_wave]

        integrated_wave = torch.cat(wave_list, dim=0)

        # p = LSD_csv_sample['P_index'].values[0] if not pd.isnull(LSD_csv_sample['P_index'].values[0]) else np.nan
        # s = LSD_csv_sample['S_index'].values[0] if not pd.isnull(LSD_csv_sample['S_index'].values[0]) else np.nan
        sample = {'data': integrated_wave,
                  'p': p,
                  's': s,
                   # 'data_path':self.csv_path,
                  # TODO:for debug
                #   "original_data":integrated_wave,
                #   "original_p":p,
                # "original_s":s
                  }
        if self.augmentations is not None:
            x = self.augmentations(sample)  # x['data']:[x_q, x_k] or [x, label]
        else:
            raise Exception()

        return x

def diting2_publish_igp(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data # TODO csv层面需要划分好训练集和验证集
    assert input_filename
    dataset = Diting2DATASET(
        json_path=input_filename,
        hdf5_path=args.hdf5_path,
        augmentations=preprocess_fn,
        sample_num=args.sample_num
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        persistent_workers=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def DiTing330km_publish(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data 
    assert input_filename
    assert input_filename.endswith('.csv'), f"DiTing330km_publish only support csv file, but got {input_filename}"
    dataset = Diting1DATASET(
        csv_path=input_filename,
        hdf5_folder_path=args.hdf5_path,
        augmentations=preprocess_fn,
        sample_num=args.sample_num,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        persistent_workers=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def PNW(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data 
    assert input_filename
    assert input_filename.endswith('.csv'), f"PNW only support csv file, but got {input_filename}"
    dataset = PNWDATASET(
        csv_path=input_filename,
        hdf5_path=args.hdf5_path,
        augmentations=preprocess_fn,
        sample_num=args.sample_num,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        persistent_workers=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def STEAD(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data 
    assert input_filename
    assert input_filename.endswith('.csv'), f"PNW only support csv file, but got {input_filename}"
    dataset = STEADDATASET(
        csv_path=input_filename,
        hdf5_path=args.hdf5_path,
        augmentations=preprocess_fn,
        sample_num=args.sample_num,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        persistent_workers=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

class STEADDATASET(Dataset):
    def __init__(self, csv_path, hdf5_path, augmentations,sample_num):
        # logging.debug(f'Loading csv data from {csv_path}.')
        # logging.debug(f'Loading h5py folder from {hdf5_path}.')
        target_columns = ['Key','P_index', 'Pn_index', 'Pg_index', 'S_index', 'Sn_index', 'Sg_index']
        self.csv = pd.read_csv(csv_path, dtype={'Key': str})[target_columns]
        if sample_num > 0:
            self.csv = self.csv.sample(n=sample_num, random_state=42)
        self.csv_path = csv_path
        print(f"STEADDATASET:{len(self.csv)}")
        self.hdf5_path = hdf5_path
        self.augmentations = augmentations
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        # logging.debug('Done loading data.')


    def __len__(self):
        return len(self.csv)
    
    def get_data(self,key):
        dataset = self.hdf5_file.get('earthquake/local/'+str(key))
        data = np.array(dataset).astype(np.float32)
        return data[:,::-1]

    def __getitem__(self, idx):
        LSD_csv_sample = self.csv.iloc[[idx]]
        columns = ['P_index', 'Pn_index', 'Pg_index']
        p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        columns = ['S_index','Sn_index', 'Sg_index']
        s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        
        key = LSD_csv_sample['Key'].iloc[0]
        # LSD_hdf5_sample = data.get_from_STEAD(key, self.hdf5_path, is_noise=False)
        LSD_hdf5_sample = self.get_data(key)
        
        
        times = 0
        while not is_valid(p,s,LSD_hdf5_sample):
            idx = random.randint(0, len(self.csv) - 1)
            LSD_csv_sample = self.csv.iloc[[idx]]
            columns = ['P_index', 'Pn_index', 'Pg_index']
            p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            columns = ['S_index','Sn_index', 'Sg_index']
            s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            
            key = LSD_csv_sample['Key'].iloc[0]
            
            LSD_hdf5_sample = self.get_data(key)
            
            times += 1
            # print(f"{self.csv_path} resample times: {times} for data invalid!")

        Z_wave = torch.from_numpy(LSD_hdf5_sample[:, 0]).reshape(1, -1).to(dtype=torch.float32) # ->[1,12000]
        N_wave = torch.from_numpy(LSD_hdf5_sample[:, 1]).reshape(1, -1).to(dtype=torch.float32)
        E_wave = torch.from_numpy(LSD_hdf5_sample[:, 2]).reshape(1, -1).to(dtype=torch.float32)

        # length = Z_wave.shape[1]

        wave_list = [Z_wave, N_wave, E_wave]

        integrated_wave = torch.cat(wave_list, dim=0)

        # p = LSD_csv_sample['P_index'].values[0] if not pd.isnull(LSD_csv_sample['P_index'].values[0]) else np.nan
        # s = LSD_csv_sample['S_index'].values[0] if not pd.isnull(LSD_csv_sample['S_index'].values[0]) else np.nan
        sample = {'data': integrated_wave,
                  'p': p,
                  's': s,
                   # 'data_path':self.csv_path,
                    # TODO:for debug
                #   "original_data":integrated_wave,
                #   "original_p":p,
                # "original_s":s,
                # "orginal_numerical":key,
                  }
        if self.augmentations is not None:
            x = self.augmentations(sample)  # x['data']:[x_q, x_k] or [x, label]
        else:
            raise Exception()

        return x


def MLAAPDE(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data 
    assert input_filename
    assert input_filename.endswith('.csv'), f"PNW only support csv file, but got {input_filename}"
    dataset = MLAAPDEDATASET(
        csv_path=input_filename,
        hdf5_path=args.hdf5_path,
        augmentations=preprocess_fn,
        sample_num=args.sample_num,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        persistent_workers=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

class MLAAPDEDATASET(Dataset):
    def __init__(self, csv_path, hdf5_path, augmentations,sample_num):
        # logging.debug(f'Loading csv data from {csv_path}.')
        # logging.debug(f'Loading h5py folder from {hdf5_path}.')
        target_columns = ['Key','P_index', 'Pn_index', 'Pg_index', 'S_index', 'Sn_index', 'Sg_index']
        self.csv = pd.read_csv(csv_path, dtype={'Key': str})[target_columns]
        if sample_num > 0:
            self.csv = self.csv.sample(n=sample_num, random_state=42)
        self.csv_path = csv_path
        print(f"MLAAPDEDATASET:{len(self.csv)}")
        self.hdf5_path = hdf5_path
        self.hdf5_files = {}
        for root, dirs, files in os.walk(hdf5_path):
            for file in files:
                if file.endswith('.h5'):
                    self.hdf5_files[os.path.join(root, file)] = h5py.File(os.path.join(root, file), 'r')
        self.augmentations = augmentations
        # logging.debug('Done loading data.')


    def __len__(self):
        return len(self.csv)
    
    def get_data(self,key):
        key_splits = key.split('|')
        cur_h5path = key_splits[0]
        event_id = key_splits[1]
        waves_id = key_splits[2]
        phase_id = key_splits[3]
        f = self.hdf5_files[os.path.join(self.hdf5_path,cur_h5path)]
        waveforms = f[event_id][waves_id][phase_id][()]
        # reshape (A, B) to (B, A)
        waveforms = np.array(waveforms, dtype=np.int32).transpose()
        # ENZ to ZNE
        waveforms = waveforms[:, ::-1]
        # resample to 100 Hz
        waveforms = resample(waveforms, 12000, axis=0)
        # taper the waveforms to avoid edge effect
        waveforms = waveforms * tukey(12000, alpha=0.05, sym=True)[:, np.newaxis]
        data = np.array(waveforms).astype(np.float32)
        return data[:,::-1]

    def __getitem__(self, idx):
        LSD_csv_sample = self.csv.iloc[[idx]]
        columns = ['P_index', 'Pn_index', 'Pg_index']
        p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        columns = ['S_index','Sn_index', 'Sg_index']
        s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        
        key = LSD_csv_sample['Key'].iloc[0]
        # LSD_hdf5_sample = data.get_from_MLAAPDE(key, self.hdf5_path)
        LSD_hdf5_sample = self.get_data(key)
        
        times = 0
        while not is_valid(p,s,LSD_hdf5_sample):
            idx = random.randint(0, len(self.csv) - 1)
            LSD_csv_sample = self.csv.iloc[[idx]]
            columns = ['P_index', 'Pn_index', 'Pg_index']
            p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            columns = ['S_index','Sn_index', 'Sg_index']
            s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            
            key = LSD_csv_sample['Key'].iloc[0]
            # LSD_hdf5_sample = data.get_from_MLAAPDE(key, self.hdf5_path)
            LSD_hdf5_sample = self.get_data(key)
            
            times += 1
            # print(f"{self.csv_path} resample times: {times} for data invalid!")

        Z_wave = torch.from_numpy(LSD_hdf5_sample[:, 0]).reshape(1, -1).to(dtype=torch.float32) # ->[1,12000]
        N_wave = torch.from_numpy(LSD_hdf5_sample[:, 1]).reshape(1, -1).to(dtype=torch.float32)
        E_wave = torch.from_numpy(LSD_hdf5_sample[:, 2]).reshape(1, -1).to(dtype=torch.float32)

        # length = Z_wave.shape[1]

        wave_list = [Z_wave, N_wave, E_wave]

        integrated_wave = torch.cat(wave_list, dim=0)

        # p = LSD_csv_sample['P_index'].values[0] if not pd.isnull(LSD_csv_sample['P_index'].values[0]) else np.nan
        # s = LSD_csv_sample['S_index'].values[0] if not pd.isnull(LSD_csv_sample['S_index'].values[0]) else np.nan
        sample = {'data': integrated_wave,
                  'p': p,
                  's': s,
                   # 'data_path':self.csv_path,
                    # TODO:for debug
                #   "original_data":integrated_wave,
                #   "original_p":p,
                # "original_s":s,
                # "orginal_numerical":key,
                  }
        if self.augmentations is not None:
            x = self.augmentations(sample)  # x['data']:[x_q, x_k] or [x, label]
        else:
            raise Exception()

        return x

def CREDITX1(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data 
    assert input_filename
    assert input_filename.endswith('.csv'), f"PNW only support csv file, but got {input_filename}"
    dataset = CREDITX1DATASET(
        csv_path=input_filename,
        hdf5_path=args.hdf5_path,
        augmentations=preprocess_fn,
        sample_num=args.sample_num,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        persistent_workers=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

class CREDITX1DATASET(Dataset):
    def __init__(self, csv_path, hdf5_path, augmentations,sample_num):
        # logging.debug(f'Loading csv data from {csv_path}.')
        # logging.debug(f'Loading h5py folder from {hdf5_path}.')
        target_columns = ['Key','P_index', 'Pn_index', 'Pg_index', 'S_index', 'Sn_index', 'Sg_index']
        self.csv = pd.read_csv(csv_path, dtype={'Key': str})[target_columns]
        if sample_num > 0:
            self.csv = self.csv.sample(n=sample_num, random_state=42)
        self.csv_path = csv_path
        print(f"CREDITX1DATASET:{len(self.csv)}")
        self.hdf5_path = hdf5_path
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.augmentations = augmentations
        # logging.debug('Done loading data.')


    def __len__(self):
        return len(self.csv)
    
    def get_data(self,key):
        t_Length = min([int(self.hdf5_file[key]['BHZ'].attrs['waveform_length_sample']), int(self.hdf5_file[key]['BHN'].attrs['waveform_length_sample']), int(self.hdf5_file[key]['BHE'].attrs['waveform_length_sample'])])
        
        waveforms = np.zeros((int(t_Length), 3), dtype=np.float32)
        waveforms[:, 0] = self.hdf5_file[key]['BHZ'][()][:t_Length]
        waveforms[:, 1] = self.hdf5_file[key]['BHN'][()][:t_Length]
        waveforms[:, 2] = self.hdf5_file[key]['BHE'][()][:t_Length]
        
        return waveforms

    def __getitem__(self, idx):
        LSD_csv_sample = self.csv.iloc[[idx]]
        columns = ['P_index', 'Pn_index', 'Pg_index']
        p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        columns = ['S_index','Sn_index', 'Sg_index']
        s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        
        key = LSD_csv_sample['Key'].iloc[0]
        # LSD_hdf5_sample = data.get_from_CREDITX1(key, self.hdf5_path)
        LSD_hdf5_sample = self.get_data(key)
        
        times = 0
        while not is_valid(p,s,LSD_hdf5_sample):
            idx = random.randint(0, len(self.csv) - 1)
            LSD_csv_sample = self.csv.iloc[[idx]]
            columns = ['P_index', 'Pn_index', 'Pg_index']
            p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            columns = ['S_index','Sn_index', 'Sg_index']
            s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            
            key = LSD_csv_sample['Key'].iloc[0]
            # LSD_hdf5_sample = data.get_from_CREDITX1(key, self.hdf5_path)
            LSD_hdf5_sample = self.get_data(key)
            
            times += 1
            # print(f"{self.csv_path} resample times: {times} for data invalid!")

        Z_wave = torch.from_numpy(LSD_hdf5_sample[:, 0]).reshape(1, -1).to(dtype=torch.float32) # ->[1,12000]
        N_wave = torch.from_numpy(LSD_hdf5_sample[:, 1]).reshape(1, -1).to(dtype=torch.float32)
        E_wave = torch.from_numpy(LSD_hdf5_sample[:, 2]).reshape(1, -1).to(dtype=torch.float32)

        # length = Z_wave.shape[1]

        wave_list = [Z_wave, N_wave, E_wave]

        integrated_wave = torch.cat(wave_list, dim=0)

        # p = LSD_csv_sample['P_index'].values[0] if not pd.isnull(LSD_csv_sample['P_index'].values[0]) else np.nan
        # s = LSD_csv_sample['S_index'].values[0] if not pd.isnull(LSD_csv_sample['S_index'].values[0]) else np.nan
        sample = {'data': integrated_wave,
                  'p': p,
                  's': s,
                   # 'data_path':self.csv_path,
                    # TODO:for debug
                #   "original_data":integrated_wave,
                #   "original_p":p,
                # "original_s":s,
                # "orginal_numerical":key,
                  }
        if self.augmentations is not None:
            x = self.augmentations(sample)  # x['data']:[x_q, x_k] or [x, label]
        else:
            raise Exception()

        return x
    
    
def INSTANCE(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data 
    assert input_filename
    assert input_filename.endswith('.csv'), f"PNW only support csv file, but got {input_filename}"
    dataset = INSTANCEDATASET(
        csv_path=input_filename,
        hdf5_path=args.hdf5_path,
        augmentations=preprocess_fn,
        sample_num=args.sample_num,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        persistent_workers=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

class INSTANCEDATASET(Dataset):
    def __init__(self, csv_path, hdf5_path, augmentations,sample_num):
        # logging.debug(f'Loading csv data from {csv_path}.')
        # logging.debug(f'Loading h5py folder from {hdf5_path}.')
        target_columns = ['Key','P_index', 'Pn_index', 'Pg_index', 'S_index', 'Sn_index', 'Sg_index']
        self.csv = pd.read_csv(csv_path, dtype={'Key': str})[target_columns]
        if sample_num > 0:
            self.csv = self.csv.sample(n=sample_num, random_state=42)
        self.csv_path = csv_path
        print(f"INSTANCEDATASET:{len(self.csv)}")
        self.hdf5_path = hdf5_path
        self.hdf5 = h5py.File(hdf5_path, 'r')
        self.augmentations = augmentations
        # logging.debug('Done loading data.')


    def __len__(self):
        return len(self.csv)
    
    def get_data(self,key):
        dataset = self.hdf5.get('data/'+str(key))
        data_t = np.array(dataset).astype(np.float32)
        data = np.zeros([12000,3])
        data[:,0] = data_t[2,:]
        data[:,1] = data_t[1,:]
        data[:,2] = data_t[0,:]
        return data

    def __getitem__(self, idx):
        LSD_csv_sample = self.csv.iloc[[idx]]
        columns = ['P_index', 'Pn_index', 'Pg_index']
        p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        columns = ['S_index','Sn_index', 'Sg_index']
        s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        
        key = LSD_csv_sample['Key'].iloc[0]
        # LSD_hdf5_sample = data.get_from_INSTANCE(key, self.hdf5_path)
        LSD_hdf5_sample = self.get_data(key)
        
        
        times = 0
        while not is_valid(p,s,LSD_hdf5_sample):
            idx = random.randint(0, len(self.csv) - 1)
            LSD_csv_sample = self.csv.iloc[[idx]]
            columns = ['P_index', 'Pn_index', 'Pg_index']
            p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            columns = ['S_index','Sn_index', 'Sg_index']
            s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            
            key = LSD_csv_sample['Key'].iloc[0]
            # LSD_hdf5_sample = data.get_from_INSTANCE(key, self.hdf5_path)
            LSD_hdf5_sample = self.get_data(key)
            
            times += 1
            # print(f"{self.csv_path} resample times: {times} for data invalid!")
            
            
        Z_wave = torch.from_numpy(LSD_hdf5_sample[:, 0]).reshape(1, -1).to(dtype=torch.float32) # ->[1,12000]
        N_wave = torch.from_numpy(LSD_hdf5_sample[:, 1]).reshape(1, -1).to(dtype=torch.float32)
        E_wave = torch.from_numpy(LSD_hdf5_sample[:, 2]).reshape(1, -1).to(dtype=torch.float32)

        # length = Z_wave.shape[1]

        wave_list = [Z_wave, N_wave, E_wave]

        integrated_wave = torch.cat(wave_list, dim=0)

        # p = LSD_csv_sample['P_index'].values[0] if not pd.isnull(LSD_csv_sample['P_index'].values[0]) else np.nan
        # s = LSD_csv_sample['S_index'].values[0] if not pd.isnull(LSD_csv_sample['S_index'].values[0]) else np.nan
        sample = {'data': integrated_wave,
                  'p': p,
                  's': s,
                   # 'data_path':self.csv_path,
                    # TODO:for debug
                #   "original_data":integrated_wave,
                #   "original_p":p,
                # "original_s":s,
                # "orginal_numerical":key,
                  }
        if self.augmentations is not None:
            x = self.augmentations(sample)  # x['data']:[x_q, x_k] or [x, label]
        else:
            raise Exception()

        return x
    

def CEA09_20(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data 
    assert input_filename
    assert input_filename.endswith('.csv'), f"PNW only support csv file, but got {input_filename}"
    dataset = CEA09_20DATASET(
        csv_path=input_filename,
        hdf5_path=args.hdf5_path,
        augmentations=preprocess_fn,
        sample_num=args.sample_num,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        persistent_workers=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

class CEA09_20DATASET(Dataset):
    def __init__(self, csv_path, hdf5_path, augmentations,sample_num):
        # logging.debug(f'Loading csv data from {csv_path}.')
        # logging.debug(f'Loading h5py folder from {hdf5_path}.')
        target_columns = ['Key','P_index', 'Pn_index', 'Pg_index', 'S_index', 'Sn_index', 'Sg_index']
        self.csv = pd.read_csv(csv_path, dtype={'Key': str})[target_columns]
        if sample_num > 0:
            self.csv = self.csv.sample(n=sample_num, random_state=42)
        self.csv_path = csv_path
        print(f"CEA09_20DATASET:{len(self.csv)}")
        self.hdf5_path = hdf5_path
        self.hdf5_files = {}
        for year in range(2009,2021):
            path = os.path.join(hdf5_path,f"h5data_{year}",str(year)+'.h5')
            self.hdf5_files[path] = h5py.File(path, 'r')
        self.augmentations = augmentations
        # logging.debug('Done loading data.')


    def __len__(self):
        return len(self.csv)
    
    def get_data(self,key):
        key_splits = key.split('|')
        cur_h5path = key_splits[0]
        t_ev_key = key_splits[1]
        t_sta_key = key_splits[2]
        t_Instrument = key_splits[3]
        f = self.hdf5_files[self.hdf5_path + cur_h5path]
        # get the waveforms
        Z_data = f[t_ev_key][t_sta_key][t_Instrument+'Z'][()]
        N_data = f[t_ev_key][t_sta_key][t_Instrument+'N'][()]
        E_data = f[t_ev_key][t_sta_key][t_Instrument+'E'][()]  
        data_length = min([len(Z_data), len(N_data), len(E_data)])
        waveforms = np.zeros((data_length, 3))
        waveforms[:,0] = Z_data[:data_length]
        waveforms[:,1] = N_data[:data_length]
        waveforms[:,2] = E_data[:data_length]
        data = np.array(waveforms).astype(np.float32)
        
        return data

    def __getitem__(self, idx):
        LSD_csv_sample = self.csv.iloc[[idx]]
        columns = ['P_index', 'Pn_index', 'Pg_index']
        p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        columns = ['S_index','Sn_index', 'Sg_index']
        s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        
        key = LSD_csv_sample['Key'].iloc[0]
        # LSD_hdf5_sample = data.get_from_CEA09_20(key, self.hdf5_path)
        LSD_hdf5_sample = self.get_data(key)
        
        
        times = 0
        while not is_valid(p,s,LSD_hdf5_sample):
            idx = random.randint(0, len(self.csv) - 1)
            LSD_csv_sample = self.csv.iloc[[idx]]
            columns = ['P_index', 'Pn_index', 'Pg_index']
            p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            columns = ['S_index','Sn_index', 'Sg_index']
            s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            
            key = LSD_csv_sample['Key'].iloc[0]
            # LSD_hdf5_sample = data.get_from_CEA09_20(key, self.hdf5_path)
            LSD_hdf5_sample = self.get_data(key)
            
            times += 1
            # print(f"{self.csv_path} resample times: {times} for data invalid!")
            
        Z_wave = torch.from_numpy(LSD_hdf5_sample[:, 0]).reshape(1, -1).to(dtype=torch.float32) # ->[1,12000]
        N_wave = torch.from_numpy(LSD_hdf5_sample[:, 1]).reshape(1, -1).to(dtype=torch.float32)
        E_wave = torch.from_numpy(LSD_hdf5_sample[:, 2]).reshape(1, -1).to(dtype=torch.float32)

        # length = Z_wave.shape[1]

        wave_list = [Z_wave, N_wave, E_wave]

        integrated_wave = torch.cat(wave_list, dim=0)

        # p = LSD_csv_sample['P_index'].values[0] if not pd.isnull(LSD_csv_sample['P_index'].values[0]) else np.nan
        # s = LSD_csv_sample['S_index'].values[0] if not pd.isnull(LSD_csv_sample['S_index'].values[0]) else np.nan
        sample = {'data': integrated_wave,
                  'p': p,
                  's': s,
                   # 'data_path':self.csv_path,
                    # TODO:for debug
                #   "original_data":integrated_wave,
                #   "original_p":p,
                # "original_s":s,
                "orginal_numerical":key,
                  }
        if self.augmentations is not None:
            x = self.augmentations(sample)  # x['data']:[x_q, x_k] or [x, label]
        else:
            raise Exception()

        return x


def CSNCD(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data 
    assert input_filename
    assert input_filename.endswith('.csv'), f"PNW only support csv file, but got {input_filename}"
    dataset = CSNCDDATASET(
        csv_path=input_filename,
        hdf5_path=args.hdf5_path,
        augmentations=preprocess_fn,
        sample_num=args.sample_num,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        persistent_workers=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

class CSNCDDATASET(Dataset):
    def __init__(self, csv_path, hdf5_path, augmentations,sample_num):
        # logging.debug(f'Loading csv data from {csv_path}.')
        # logging.debug(f'Loading h5py folder from {hdf5_path}.')
        target_columns = ['Key','P_index', 'Pn_index', 'Pg_index', 'S_index', 'Sn_index', 'Sg_index']
        self.csv = pd.read_csv(csv_path, dtype={'Key': str})[target_columns]
        if sample_num > 0:
            self.csv = self.csv.sample(n=sample_num, random_state=42)
        self.csv_path = csv_path
        print(f"CSNCDDATASET:{len(self.csv)}")
        self.hdf5_path = hdf5_path
        self.hdf5_files = {}
        for year in range(2009,2023):
            self.hdf5_files[hdf5_path + f"{year}.ayr.h5"] = h5py.File(hdf5_path + f"{year}.ayr.h5", 'r')
        self.augmentations = augmentations
        # logging.debug('Done loading data.')


    def __len__(self):
        return len(self.csv)
    
    def get_data(self,key):
        key_splits = key.split('|')
        cur_h5_path = key_splits[0]
        ev_key = key_splits[1]
        sta_key = key_splits[2]
        t_Instrument = key_splits[3]
        cur_h5_name = self.hdf5_path + cur_h5_path
        try:
            f = self.hdf5_files[cur_h5_name]
            Z_data = f[ev_key][sta_key][t_Instrument+'Z'][()]
            N_data = f[ev_key][sta_key][t_Instrument+'N'][()]
            E_data = f[ev_key][sta_key][t_Instrument+'E'][()]
            data_length = min([len(Z_data), len(N_data), len(E_data)])
            waveforms = np.zeros((data_length*2, 3))
            waveforms[:,0] = resample(Z_data[:data_length], data_length*2)
            waveforms[:,1] = resample(N_data[:data_length], data_length*2)
            waveforms[:,2] = resample(E_data[:data_length], data_length*2)
        except ValueError:
            return None
        return waveforms

    def __getitem__(self, idx):
        LSD_csv_sample = self.csv.iloc[[idx]]
        columns = ['P_index', 'Pn_index', 'Pg_index']
        p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        columns = ['S_index','Sn_index', 'Sg_index']
        s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
        
        key = LSD_csv_sample['Key'].iloc[0]
        # LSD_hdf5_sample = data.get_from_CSNCD(key, self.hdf5_path)
        LSD_hdf5_sample = self.get_data(key)
        
        times = 0
        while not is_valid(p,s,LSD_hdf5_sample):
            idx = random.randint(0, len(self.csv) - 1)
            LSD_csv_sample = self.csv.iloc[[idx]]
            columns = ['P_index', 'Pn_index', 'Pg_index']
            p = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            columns = ['S_index','Sn_index', 'Sg_index']
            s = next((LSD_csv_sample[column].iloc[0] for column in columns if not pd.isnull(LSD_csv_sample[column].iloc[0])), np.nan)
            
            key = LSD_csv_sample['Key'].iloc[0]
            # LSD_hdf5_sample = data.get_from_CSNCD(key, self.hdf5_path)
            LSD_hdf5_sample = self.get_data(key)
            
            times += 1
            # print(f"{self.csv_path} resample times: {times} for data invalid!")
        

        Z_wave = torch.from_numpy(LSD_hdf5_sample[:, 0]).reshape(1, -1).to(dtype=torch.float32) # ->[1,12000]
        N_wave = torch.from_numpy(LSD_hdf5_sample[:, 1]).reshape(1, -1).to(dtype=torch.float32)
        E_wave = torch.from_numpy(LSD_hdf5_sample[:, 2]).reshape(1, -1).to(dtype=torch.float32)

        # length = Z_wave.shape[1]

        wave_list = [Z_wave, N_wave, E_wave]

        integrated_wave = torch.cat(wave_list, dim=0)

        # p = LSD_csv_sample['P_index'].values[0] if not pd.isnull(LSD_csv_sample['P_index'].values[0]) else np.nan
        # s = LSD_csv_sample['S_index'].values[0] if not pd.isnull(LSD_csv_sample['S_index'].values[0]) else np.nan
        sample = {'data': integrated_wave,
                  'p': p,
                  's': s,
                   # 'data_path':self.csv_path,
                    # TODO:for debug
                #   "original_data":integrated_wave,
                #   "original_p":p,
                # "original_s":s,
                "orginal_numerical":key,
                  }
        if self.augmentations is not None:
            x = self.augmentations(sample)  # x['data']:[x_q, x_k] or [x, label]
        else:
            raise Exception()

        return x
