import pandas as pd
from typing import Optional,Tuple
import copy

from typing import Optional, Tuple
import os
import pandas as pd
import numpy as np
from operator import itemgetter
import h5py
from finetuneing.utils import cal_snr


class DatasetBase:
    """
    The base class for datasets.
    """

    _name: str
    _part_range: Optional[tuple]
    _channels: list
    _sampling_rate: int

    def __init__(
        self,
        seed: int,
        mode: str,
        data_dir: str,
        shuffle: bool = True,
        data_split: bool = True,
        train_size: float = 0.8,
        val_size: float = 0.1,
        downstream_task = 'dis',
        sample_num = -1,
    ):
        """
        Args:
            seed: int
                Random seed.
            mode: str
                train / val /test
            data_dir: str
                Directory of dataset.
            shuffle: bool
                If true, meta data will be shuffled.
            data_split: bool
                whether split dataset to train/val/test
            train_size: float
                size of train set
            val_size: float
                size of validation set
        """
        self._seed = seed

        assert mode.lower() in ["train", "val", "test"]
        self._mode = mode.lower()
 
        self._data_dir = data_dir
        self._shuffle = shuffle
        self._data_split = data_split
        self.sample_num = sample_num
        
        
        assert (
            train_size + val_size <= 1.0
        ), f"train_size:{train_size}, val_size:{val_size}"
        self._train_size = train_size
        self._val_size = val_size
        self.downstream_task = downstream_task
        self._meta_data = self._load_meta_data()
    

    def _load_meta_data(self, filename = None) -> pd.DataFrame:
        pass

    def _load_event_data(self, idx: int) -> dict:
        pass 
    
    def __repr__(self) -> str:
        return (
            f"Dataset(name:{self._name}, part_range:{self._part_range}, channels:{self._channels}, "
            f"sampling_rate:{self._sampling_rate}, data_dir:{self._data_dir}, shuffle:{self._shuffle}, "
            f"data_split:{self._data_split}, train_size:{self._train_size}, val_size:{self._val_size})"
        )

    def __len__(self):
        return len(self._meta_data)
    
    def __getitem__(self,idx:int)->Tuple[dict,dict]:
        return self._load_event_data(idx=idx)

    @classmethod
    def name(cls):
        return cls._name

    @classmethod
    def sampling_rate(cls):
        return cls._sampling_rate

    @classmethod
    def channels(cls):
        return copy.deepcopy(cls._channels)


class mixDataset(DatasetBase):
    """mixDataset"""

    _name = "mixDataset"
    _channels = ["z", "n", "e"]
    _part_range = None
    _sampling_rate = 100

    def __init__(
            self,
            seed: int,
            mode: str,
            data_dir: str,
            shuffle: bool = True,
            data_split: bool = True,
            train_size: float = 0.8,
            val_size: float = 0.1,
            downstream_task='dis',
            **kwargs
    ):
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir, # csv
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
            downstream_task=downstream_task,
            sample_num=kwargs['sample_num'],
        )
        path = os.path.join(self._data_dir, f"LSD_small_waveform_data.hdf5")
        self.LSD_hdf5 = h5py.File(path, 'r')


    def _load_meta_data(self, filename=None) -> pd.DataFrame:
        meta_df = pd.read_csv(self._data_dir, dtype={'Key': str})
        if self.sample_num > 0:
            meta_df = meta_df.sample(n=self.sample_num, random_state=42)

        
        meta_df = meta_df.dropna(subset=['Dis'])
        meta_df = meta_df[meta_df['Dis'] < 500]
        if self._mode == "test":
            if self.downstream_task == 'dis':
                condition = (pd.isna(meta_df['P_index']) & pd.isna(meta_df['Pn_index']) & pd.isna(meta_df['Pg_index']))
                meta_df = meta_df[~condition]
                print("[dis|test]p_index|Pn_index|Pg_index is null:",(condition==1).sum())
                print("[dis|test]samples num:",len(meta_df))
            elif 'emg' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Mag_value'])
            elif 'baz' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Baz'])
            elif 'fmp' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['P_polarity'])
            elif 'dep' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Eq_depth'])
            elif 'cls' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Type'])
            elif self.downstream_task == 'dpk':
                meta_df = meta_df.dropna(subset=['P_index'])
                meta_df = meta_df.dropna(subset=['S_index'])
            else:
                raise NotImplementedError(f"Downstream task {self.downstream_task} not implemented")
            return meta_df
        
        elif self._mode == "train":
            if self.downstream_task == 'dis':
                condition = (pd.isna(meta_df['P_index']) & pd.isna(meta_df['Pn_index']) & pd.isna(meta_df['Pg_index']))
                meta_df = meta_df[~condition]
                print("[train]p_index|Pn_index|Pg_index is null:",(condition==1).sum())
                print("[train]samples num:",len(meta_df))
            elif 'emg' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Mag_value'])
            elif 'baz' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Baz'])
            elif 'fmp' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['P_polarity'])
            elif 'dep' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Eq_depth'])
            elif 'cls' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Type'])
            elif self.downstream_task == 'dpk':
                meta_df = meta_df.dropna(subset=['P_index'])
                meta_df = meta_df.dropna(subset=['S_index'])
            else:
                raise NotImplementedError(f"Downstream task {self.downstream_task} not implemented")

            if self._shuffle:
                meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)

            if self._data_split:
                irange = {}
                irange["train"] = [0, int(self._train_size * meta_df.shape[0])]
                irange["val"] = [irange["train"][1], meta_df.shape[0]]

                r = irange[self._mode]
                meta_df = meta_df.iloc[r[0]: r[1], :]

            return meta_df

    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        """Load event data(__getitem__)

        Args:
            idx (int): Index.

        Raises:
            ValueError: Unknown 'mag_type'

        Returns:
            dict: Data of event.
            dict: Meta data.
        """

        target_event = self._meta_data.iloc[idx]
        key = target_event["Key"]

        # parent_dir = os.path.dirname(self._data_dir)

        dataset = self.LSD_hdf5[str(key)]
        data = np.array(dataset).astype(np.float32).T
        # (
        #     ppk,
        #     spk,
        #     motion,
        #     baz,
        #     dis,
        #     evmag,
        #     dep,
        #     cls
        # ) = itemgetter(
        #     "P_index",
        #     "S_index",
        #     "P_polarity",
        #     "Baz",
        #     "Dis",
        #     "Mag_value",
        #     "Eq_depth",
        #     "Type"
        # )(
        #     target_event
        # )
        def get_index(event, *keys):
            for key in keys:
                if key in event and not pd.isna(event[key]):
                    return event[key]
            raise ValueError(f"{keys} not found in {event}")
                

        # 使用辅助函数来获取P_index，如果P_index不存在，则尝试Pn_index，再不行则Pg_index
        ppk = get_index(target_event, "P_index", "Pn_index", "Pg_index")
        spk = target_event["S_index"]
        motion = target_event["P_polarity"]
        baz = target_event["Baz"]
        dis = target_event["Dis"]
        evmag = target_event["Mag_value"]
        dep = target_event["Eq_depth"]
        cls = target_event["Type"]

        if pd.notnull(motion) and motion.lower() not in ["", "n"]:
            motion = {"up": 0, "c": 0, "r": 1, "down": 1}[motion.lower()]

        if pd.notnull(baz):
            baz = baz % 360

        # Type: Eq, Noise, Non-Natural
        # if Type equals to Eq, then it is an earthquake event
        # if Type equals to Noise, then it is a noise event
        # else it is a non-natural event
        if cls == 'Eq':
            cls = 0
        elif cls == 'Noise':
            cls = 1
        else:
            cls = 2

        event = {
            "data": data,
            "ppks": [int(ppk)] if pd.notnull(ppk) else [],
            "spks": [int(spk)] if pd.notnull(spk) else [],
            "emg": [evmag] if pd.notnull(evmag) else [],
            "fmp": [motion] if pd.notnull(motion) else [],
            "baz": [baz] if pd.notnull(baz) else [],
            "dis": [dis] if pd.notnull(dis) else [],
            "dep": [dep] if pd.notnull(dep) else [],
            "cls": [cls] if pd.notnull(cls) else [],
            "snr": np.array(cal_snr(data=data, pat=ppk)) if ppk > 0 else 0.
        }

        return event