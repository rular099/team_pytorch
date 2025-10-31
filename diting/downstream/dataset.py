from finetuneing.datasets.SFTData import SFTData
from finetuneing.training.preprocess import SFTDataset

class SFTDataCustom(SFTData):
    """SFTData Dataset"""

    _name = "SFTData"
    _channels = ["z", "n", "e"]
    _part_range = None
    _sampling_rate = 100

    def __init__(
            self,
            seed: int,
            mode: str,
            data_dir: str, # hdf5 train folder
            meta_data_path: str, # metadata path
            shuffle: bool = True,
            data_split: bool = False,
            train_size: float = 0.8,
            val_size: float = 0.1,
            downstream_task='dis',
            target_column='Dis',
            # subset names (csv string)
            subset_names: str = None,
            **kwargs
    ):
        self.data_dir = data_dir
        self.meta_data_path = meta_data_path
        self.sample_num = kwargs['sample_num']
        self.downstream_task = downstream_task
        self.target_column = target_column
        if mode == 'test':
            self.test_p_position = kwargs['test_p_postion_ratio']
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
            downstream_task=downstream_task,
            subset_names = subset_names,
        )
        if self._mode == 'train' or self._mode == 'val':
            self.hdf5_file = {}
            include_subset_names = [
                'diting1', 'DitingV2', "CSNCD-compressed", "CEA0920", "STEAD", "INSTANCE", "credit", "mlaapde", "PNW-Exotic"
            ]
            for subset_name in self.subset_names.split(";"):
                assert subset_name in include_subset_names, f"subset_name {subset_name} not in {include_subset_names}"
                self._load_hdf5_files(subset_name)
        elif self._mode == 'test':
            self.hdf5_file = data_dir
        else:
            raise NotImplementedError(f"Mode {self._mode} not implemented")

    def _load_hdf5_files(self, subset_name):
        """Load HDF5 files based on the subset name."""
        if subset_name == 'diting1':
            self.hdf5_file['diting1'] = [
                os.path.join(self.data_dir, 'DiTing330km_publish', f'DiTing330km_part_{part}.hdf5') 
                for part in range(28)
            ]
        elif subset_name == 'DitingV2':
            self.hdf5_file['diting2'] = {
                f"DiTing_{year}_{year + 1}.hdf5": 
                    os.path.join(self.data_dir, 'diting2.0_publish_igp', f"DiTing_{year}_{year + 1}.hdf5") 
                    for year in range(2020, 2023)
            }
        elif subset_name == 'CSNCD-compressed':
            self.hdf5_file['csncd'] = {}
            self.csncd_hdf5_path = os.path.join(self.data_dir,'CSNCD_compressed')
            # debug =================================================================================================
            for year in range(2009,2023):
                self.hdf5_file['csncd'][os.path.join(self.csncd_hdf5_path, f"{year}.ayr.h5")] = os.path.join(self.csncd_hdf5_path, f"{year}.ayr.h5")
            # =================================================================================================
        elif subset_name == 'CEA0920':
            self.hdf5_file['cea09_20'] = {}
            self.cea_hdf5_path = os.path.join(self.data_dir,'h5data_2009_2020')
            for year in range(2009,2021): # TODO
                path = os.path.join(self.cea_hdf5_path,f"h5data_{year}",str(year)+'.h5')
                self.hdf5_file['cea09_20'][path] = path
        elif subset_name == 'STEAD':
            self.hdf5_file['stead'] = os.path.join(self.data_dir,'STEAD/waveforms.hdf5')
        elif subset_name == 'INSTANCE':
            self.hdf5_file['instance'] = os.path.join(self.data_dir,'INSTANCE/Instance_events_counts.hdf5')
        elif subset_name == 'credit':
            self.hdf5_file['credit'] = os.path.join(self.data_dir,'CREDITX1/credit-x1.h5')
        elif subset_name == 'mlaapde':
            self.hdf5_file['mlaapde'] = {}
            self.mlaapde_hdf5_path = os.path.join(self.data_dir,'MLAAPDE')
            for root, dirs, files in os.walk(self.mlaapde_hdf5_path):
                for file in files:
                    if file.endswith('.h5'):
                        self.hdf5_file['mlaapde'][os.path.join(root, file)] = os.path.join(root, file)
        elif subset_name == 'PNW-Exotic':
            self.hdf5_file['pnw'] = os.path.join(self.data_dir,'PNW/exotic_waveforms.hdf')

    def _load_meta_data(self, filename=None) -> pd.DataFrame:
        if self._mode == "test":
            meta_df = pd.read_csv(self.meta_data_path)
            
            for label in ['P_index','Pn_index','Pg_index',
                          'S_index','Sn_index','Sg_index',
                          'Dis','Mag_value']:
                meta_df[label] = pd.to_numeric(meta_df[label], errors='coerce')
            
            meta_df['p_target'] = np.nanmin(meta_df[['P_index','Pn_index','Pg_index']].values,axis=1)
            meta_df['s_target'] = np.nanmin(meta_df[['S_index','Sn_index','Sg_index']].values,axis=1)
            
            if 'dis' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Dis'])
                meta_df = meta_df[meta_df['Dis'] < 500]
                meta_df = meta_df.dropna(subset=['p_target'])
            elif 'emg' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Mag_value'])
                meta_df = meta_df.dropna(subset=['p_target'])
                meta_df = meta_df[meta_df['Dis'] < 500]
            # elif 'det' in self.downstream_task:
            #     meta_df = meta_df.dropna(subset=['p_target'])
            #     meta_df = meta_df.dropna(subset=['s_target'])
            # elif 'ppk' in self.downstream_task:
            #     meta_df = meta_df.dropna(subset=['p_target'])
            # elif 'spk' in self.downstream_task:
            #     meta_df = meta_df.dropna(subset=['s_target'])
            elif self.target_column != "":
                meta_df = meta_df.dropna(subset=[self.target_column])
            else:
                raise NotImplementedError(f"Downstream task {self.downstream_task} not implemented")
            
            # debug ===================================================
            # meta_df = meta_df.sample(frac=0.2, replace=False, random_state=self._seed)[:100]
            # =========================================================
            meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)
            meta_df['test_p_postion_ratio'] = self.test_p_position
            print("test data describe:")
            print(meta_df.describe())
            
            return meta_df
        
        meta_df = pd.read_csv(
            self.meta_data_path,dtype={
                'Key':str,'From':str,'Dis':float,'Mag_value':float,'p_target':float,'s_target':float
            }
        )
        meta_df_list = []
        for subset_name in self.subset_names.split(";"):
            meta_df_list.append(meta_df[meta_df['From'].str.split('_').str.get(0).str.lower() == subset_name.lower()])
            
        meta_df = pd.concat(meta_df_list)
        print("the lenght of data", len(meta_df))
        meta_df = meta_df.sample(self.sample_num, random_state=self._seed)

        if self._shuffle:
            meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)

        if self._data_split:
            irange = {}
            irange["train"] = [0, int(self._train_size * meta_df.shape[0])]
            irange["val"] = [irange["train"][1], meta_df.shape[0]]

            r = irange[self._mode]
            meta_df = meta_df.iloc[r[0]: r[1], :]

        # 打印meta_df的信息
        print(meta_df.describe())
        return meta_df
    
    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        """Load evnet data

        Args:
            idx (int): Index.

        Raises:
            ValueError: Unknown 'mag_type'

        Returns:
            dict: Data of event.
            dict: Meta data.
        """
        target_event = self._meta_data.iloc[idx]
        #print('target_event is: ',target_event)
        data = self.get_data(target_event).T
        resample_times = 0
        while not is_valid(data,target_event) or (not is_dpk_valid(data,target_event)):
            idx = random.randint(0, len(self._meta_data) - 1)
            target_event = self._meta_data.iloc[idx]
            data = self.get_data(target_event).T
            print(f"resample_times:{resample_times}")
            resample_times += 1

        ppk = target_event["p_target"]
        spk = target_event["s_target"]
        dis = target_event["Dis"]
        evmag = target_event["Mag_value"]
        if self.target_column != "":
            data_label = target_event[self.target_column]

        event = {
            "data": data,
            "ppks": [int(ppk)] if pd.notnull(ppk) else [],
            "spks": [int(spk)] if pd.notnull(spk) else [],
            "emg": [evmag] if pd.notnull(evmag) else [-1],
            "dis": [dis] if pd.notnull(dis) else [-1],
            "snr": np.array(cal_snr(data=data, pat=ppk)) if ppk > 0 else 0.,
            "from":target_event["From"],
            "test_p_postion_ratio": target_event["test_p_postion_ratio"] if self._mode == 'test' else None,
            "p_exits": [1] if pd.notnull(ppk) else [0],
            "s_exits": [1] if pd.notnull(spk) else [0],
        }
        if self.target_column != "":
            event[self.downstream_task] = [data_label] if pd.notnull(data_label) else [-1],
        
        return event,target_event.to_dict()
    
    def get_credit_data(self,key):
        file_path  = self.hdf5_file['credit']
        with h5py.File(file_path, 'r') as hdf5_file:
            t_Length = min([
                int(hdf5_file[key]['BHZ'].attrs['waveform_length_sample']), 
                int(hdf5_file[key]['BHN'].attrs['waveform_length_sample']), 
                int(hdf5_file[key]['BHE'].attrs['waveform_length_sample'])
                ])
            waveforms = np.zeros((int(t_Length), 3), dtype=np.float32)
            waveforms[:, 0] = hdf5_file[key]['BHZ'][()][:t_Length]
            waveforms[:, 1] = hdf5_file[key]['BHN'][()][:t_Length]
            waveforms[:, 2] = hdf5_file[key]['BHE'][()][:t_Length]
        
        return waveforms
    
    def get_mlaapde_data(self,key):
        key_splits = key.split('|')
        cur_h5path = key_splits[0]
        event_id = key_splits[1]
        waves_id = key_splits[2]
        phase_id = key_splits[3]
        file_path = os.path.join(self.mlaapde_hdf5_path, cur_h5path)
        
        f = self.hdf5_file['mlaapde'][os.path.join(self.mlaapde_hdf5_path,cur_h5path)]
        with h5py.File(file_path, 'r') as f:
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
    
    def get_pnw_data(self,key):
        bucket, array = key.split('$')
        x, y, z = iter([int(i) for i in array.split(',:')])
        file_path = self.hdf5_file['pnw']
        with h5py.File(file_path, 'r') as f:
            LSD_hdf5_sample = np.array(f['pnw']['data'][bucket][x, :y, :z].transpose()[:, ::-1])
        return LSD_hdf5_sample
    
    def get_diting1_data(self,key,part):
        file_path = self.hdf5_file['diting1'][part]
        with h5py.File(file_path, 'r') as f:
            dataset = f.get('earthquake/'+str(key))
            data = np.array(dataset).astype(np.float32)
        return data
    
    def get_csncd_data(self,key):
        key_splits = key.split('|')
        cur_h5_path = key_splits[0]
        ev_key = key_splits[1]
        sta_key = key_splits[2]
        t_Instrument = key_splits[3]
        
        file_path = os.path.join(self.csncd_hdf5_path,cur_h5_path)
        with h5py.File(file_path, 'r') as f:
            Z_data = f[ev_key][sta_key][t_Instrument+'Z'][()]
            N_data = f[ev_key][sta_key][t_Instrument+'N'][()]
            E_data = f[ev_key][sta_key][t_Instrument+'E'][()]
            data_length = min([len(Z_data), len(N_data), len(E_data)])
            waveforms = np.zeros((data_length*2, 3))
            waveforms[:,0] = resample(Z_data[:data_length], data_length*2)
            waveforms[:,1] = resample(N_data[:data_length], data_length*2)
            waveforms[:,2] = resample(E_data[:data_length], data_length*2)
            data = waveforms.astype(np.float32)
        return data
    
    def get_cea_data(self,key):
        key_splits = key.split('|')
        cur_h5path = key_splits[0]
        t_ev_key = key_splits[1]
        t_sta_key = key_splits[2]
        t_Instrument = key_splits[3]
        cur_h5file_path = os.path.join(self.cea_hdf5_path, cur_h5path)
        with h5py.File(cur_h5file_path, 'r') as f:
            # get the waveforms
            Z_data = f[t_ev_key][t_sta_key][t_Instrument+'Z'][()]
            N_data = f[t_ev_key][t_sta_key][t_Instrument+'N'][()]
            E_data = f[t_ev_key][t_sta_key][t_Instrument+'E'][()]  
            data_length = min([len(Z_data), len(N_data), len(E_data)])
            waveforms = np.zeros((data_length, 3))
            waveforms[:,0] = Z_data[:data_length]
            waveforms[:,1] = N_data[:data_length]
            waveforms[:,2] = E_data[:data_length]
            data = waveforms.astype(np.float32)
        return data

    def get_instance_data(self,key):
        file_path = self.hdf5_file['instance']
        with h5py.File(file_path, 'r') as f:
            dataset = f.get('data/'+str(key))
            data_t = np.array(dataset).astype(np.float32)
            data = np.zeros([12000,3])
            data[:,0] = data_t[2,:]
            data[:,1] = data_t[1,:]
            data[:,2] = data_t[0,:]
        return data
    
    def get_stead_data(self,key):
        file_path = self.hdf5_file['stead']
        with h5py.File(file_path, 'r') as f:
            dataset = f.get('earthquake/local/'+str(key))
            data = np.array(dataset).astype(np.float32)
            try:
                data = data[:,::-1]
            except:
                print("stead:",data.shape)
        return data
    
    def get_data(self,target_event):
        key = target_event["Key"]
        data_name = target_event["From"]
        if data_name.startswith('diting1'):
            part = int(data_name.split('_')[-1])
            key_correct = key.split('.')
            key = key_correct[0].rjust(6,'0') + '.' + key_correct[1].ljust(4,'0')
            data = self.get_diting1_data(key,part)
        elif data_name.startswith('DitingV2'):
            year = int(data_name.split('_')[-1])
            file_path = self.hdf5_file['diting2'][f"DiTing_{year}_{year + 1}.hdf5"]
            with h5py.File(file_path, 'r') as f:
                data = f.get('earthquake').get(key)[()]
                data = np.array(data).astype(np.float32)
        elif data_name == 'CSNCD_compressed':
            data = self.get_csncd_data(key)
        elif data_name == 'CEA09_20':
            data = self.get_cea_data(key)
        elif data_name == 'instance':
            data = self.get_instance_data(key)
        elif data_name == 'stead':
            data = self.get_stead_data(key)
        elif data_name == 'credit':
            data = self.get_credit_data(key)
        elif data_name == 'mlaapde':
            data = self.get_mlaapde_data(key)
        elif data_name == 'PNW-Exotic':
            data = self.get_pnw_data(key)
        elif data_name == 'DiTingV3_Test':
            with h5py.File(self.hdf5_file, 'r') as f:
                data = np.array(f[str(key)]).astype(np.float32)
        else:
            raise NotImplementedError(f"Data name {data_name} not implemented")
            
        return data

class SFTDataset(Dataset):
    """
    Read and preprocess data.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        input_names: list,
        label_names: list,
        task_names: list,
        mode: str,
    ) -> None:
        """
        Args:
            args:argparse.Namespace
                Input arguments.
            input_names: list
                Input names. See :class:`~SeisT.config.Config` for more details.
            label_names: list
                Label names. See :class:`~SeisT.config.Config` for more details.
            task_names: list
                Task names. See :class:`~SeisT.config.Config` for more details.
            mode: str
                train/val/test.
        """

        self._seed = int(args.seed)
        self._mode = mode.lower()
        self._input_names = input_names
        self._label_names = label_names
        self._task_names = task_names
        self._max_event_num = args.max_event_num

        self._augmentation = args.augmentation and self._mode == "train"
        if self._augmentation != args.augmentation:
            print(f"[{self._mode}]Augmentation -> {self._augmentation}")

        # Dataset
        if self._mode == "train" or self._mode == 'val':
            meta_data_path = args.train_meta_data_path
            data_dir = args.train_data_dir
            sample_num = args.train_sample_num
            test_p_postion_ratio = None
        elif self._mode == 'test':
            meta_data_path = args.test_meta_data_path
            data_dir = args.test_data_dir
            sample_num = None
            test_p_postion_ratio = args.test_p_postion_ratio
        else:
            raise NotImplementedError
        
        if args.splitPS:
            self._dataset = SFTDataSplitPS(
                seed=self._seed,
                mode=self._mode,
                data_dir=data_dir,
                meta_data_path=meta_data_path,
                shuffle=args.shuffle,
                data_split=args.data_split,
                train_size=args.train_size,
                val_size=args.val_size,
                downstream_task=args.downstream_task,
                subset_names=args.subset_names,
                sample_num=sample_num,
            )
        else:
            self._dataset = SFTData(
                seed=self._seed,
                mode=self._mode,
                data_dir=data_dir,
                meta_data_path=meta_data_path,
                shuffle=args.shuffle,
                data_split=args.data_split,
                train_size=args.train_size,
                val_size=args.val_size,
                downstream_task=args.downstream_task,
                target_column=args.target_column,
                subset_names=args.subset_names,
                sample_num=sample_num,
                test_p_postion_ratio=test_p_postion_ratio,
            )
        print(self._dataset)

        self._dataset_size = len(self._dataset)

        if self._augmentation:
            print(
                f"Data augmentation: Dataset size -> {self._dataset_size}"
            )

        # Preprocessor
        self._preprocessor = DataPreprocessor(
            data_channels=self._dataset.channels(),
            sampling_rate=self._dataset.sampling_rate(),
            in_samples=args.in_samples,
            min_snr=args.min_snr,
            coda_ratio=args.coda_ratio,
            norm_mode=args.norm_mode,
            p_position_ratio=args.p_position_ratio,
            add_event_rate=args.add_event_rate,
            add_noise_rate=args.add_noise_rate,
            add_gap_rate=args.add_gap_rate,
            drop_channel_rate=args.drop_channel_rate,
            scale_amplitude_rate=args.scale_amplitude_rate,
            pre_emphasis_rate=args.pre_emphasis_rate,
            pre_emphasis_ratio=args.pre_emphasis_ratio,
            max_event_num=args.max_event_num,
            generate_noise_rate=args.generate_noise_rate,
            shift_event_rate=args.shift_event_rate,
            mask_percent=args.mask_percent,
            noise_percent=args.noise_percent,
            min_event_gap_sec=args.min_event_gap,
            soft_label_shape=args.label_shape,
            soft_label_width=int(args.label_width * self._dataset.sampling_rate()),
            operations=args.operations,
            add_mask_window_rate=args.add_mask_window_rate,
            add_noise_window_rate=args.add_noise_window_rate,
            noise_type=args.noise_type,
            whether_fixed_mask_percent=args.whether_fixed_mask_percent,
            lower_bound_mask_percent=args.lower_bound_mask_percent,
            upper_bound_mask_percent=args.upper_bound_mask_percent,
            whether_fixed_noise_percent=args.whether_fixed_noise_percent,
            lower_bound_noise_percent=args.lower_bound_noise_percent,
            upper_bound_noise_percent=args.upper_bound_noise_percent,
            shift_event_distance_percent=args.shift_event_distance_percent,
            dtype=np.float32,
            p_position_ratio_type=args.p_position_ratio_type,
            p_position_ratio_range_or_sigma=args.p_position_ratio_range_or_sigma,
        )
        self.no_event_p=args.no_event_p
        self.random_crop_p=args.random_crop_p
        self.no_event_label=args.no_event_label
        self.default_label_dis=args.default_label_dis

    def sampling_rate(self):
        return self._dataset.sampling_rate()

    def data_channels(self):
        return self._dataset.channels()
    
    def name(self):
        return f"{self._dataset.name()}_{self._mode}"

    def __len__(self) -> int:
        return self._dataset_size

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: inputs, loss_targets, metrics_targets, meta_data
        """

        # Load data
        event,meta_data = self._dataset[idx % self._dataset_size]

        # Sample window
        if self._mode == 'test':
            self._preprocessor.p_position_ratio = event['test_p_postion_ratio']
            self._preprocessor.p_position_ratio_type = 'uniform'
            self._preprocessor.p_position_ratio_range_or_sigma = 0
        else:
            if random.random() <= self.random_crop_p or event['p_exits'] == [0]:
                self._preprocessor.p_position_ratio = -1
            else:
                self._preprocessor.p_position_ratio = 0.5
                self._preprocessor.p_position_ratio_type = 'uniform'
                self._preprocessor.p_position_ratio_range_or_sigma = 0.5

        # Preprocess
        event = self._preprocessor.process(
            event=event, augmentation=self._augmentation
        )

        if len(event['ppks']) < 1:
            if self.no_event_label == 'default':
                event['emg'] = [-3.0]
                event['dis'] = [self.default_label_dis]
            else:
                event['emg'] = [-1]
                event['dis'] = [-1]

        # Generate inputs
        inputs = self._preprocessor.get_inputs(
            event=event, input_names=self._input_names
        )

        # add by zb, use 1 second before and 1 seconds after the P wave to predict the first motion polarity
        sampling_rate=self._dataset.sampling_rate(),
        if len(self._task_names) == 1 and self._task_names[0] == "fmp":
            if len(event["ppks"]) != 0:
                crop_start = max(event["ppks"][0] - int(1.0 * self._dataset.sampling_rate()), 0)
                crop_end = event["ppks"][0] + int(1.0 * self._dataset.sampling_rate()) + 1
                event["data"][:,:crop_start] = 0.
                event["data"][:,crop_end:] = 0.
        # end add by zb

        # Generate labels
        if len(self._label_names) > 1:
            loss_targets = [
                self._preprocessor.get_targets_for_loss(event=event, label_names=label_name) 
                for label_name in self._label_names
            ]
            metrics_targets = [
                self._preprocessor.get_targets_for_metrics(event=event, task_names=task_name, max_event_num=self._max_event_num)
                for task_name in self._task_names
            ]
            if 'ppks_type' in event.keys() and 'spks_type' in event.keys():
                print('hello')
                metrics_targets[0]['ppks_type'] = np.array([event['ppks_type']])
                metrics_targets[0]['spks_type'] = np.array([event['spks_type']])
        # add by lhl
        elif isinstance(self._label_names, list) and len(self._label_names) == 1 and isinstance(self._label_names[0], list): # for multi det_ppk_spk or det_ppk_spk_dis_emg
            loss_targets = {
                label_name:self._preprocessor.get_targets_for_loss(event=event, label_names=[label_name]) 
                for label_name in self._label_names[0]
            }
            metrics_targets = self._preprocessor.get_targets_for_metrics(event=event, task_names=self._task_names, max_event_num=self._max_event_num)

            '''
            metrics_targets['ppk'] = [-1e7]: p exits but shift out of window
                                   = [-1]: p not exits
            metrics_targets['det'] = [1,0]: no event
                                   = [-1,-1]: no label
            '''
            if event['p_exits'] == [0]:
                metrics_targets['ppk'] = np.array([-1])
                metrics_targets['det'] = np.array([-1,-1])
            if event['s_exits'] == [0]:
                metrics_targets['spk'] = np.array([-1])
                metrics_targets['det'] = np.array([-1,-1])
        
        elif isinstance(self._label_names, list) and len(self._label_names) == 1 : # for single dis/emg task
            loss_targets = {self._label_names[0]:
                self._preprocessor.get_targets_for_loss(
                event=event, label_names=self._label_names)
            }
            metrics_targets = self._preprocessor.get_targets_for_metrics(
                event=event, task_names=self._task_names, max_event_num=self._max_event_num
            )
            if 'ppks_type' in event.keys() and 'spks_type' in event.keys():
                metrics_targets['ppks_type'] = np.array([event['ppks_type']])
                metrics_targets['spks_type'] = np.array([event['spks_type']])
        
        # add by lhl         
        # # task_mask
        # # note: task order in `task_mask` can be found in help_builder.py `get_labels_tasks` function
        # # here will filter some labels which is affected by the shift of ppk and spk & invalid labels
        # task_mask = np.zeros(len(self._task_names), dtype=bool)
        # p_exist = (metrics_targets["ppk"] != int(-1e7)).any()
        # s_exist = (metrics_targets["spk"] != int(-1e7)).any()
        # for i, task in enumerate(self._task_names):
        #     if task == "ppk" and p_exist:
        #         task_mask[i] = 1
            
        #     elif task == "spk" and s_exist:
        #         task_mask[i] = 1

        #     elif task == "det":
        #         assert len(metrics_targets["det"]) == 2, f"only support 1 event, but det:{metrics_targets['det']}"
        #         if metrics_targets[task][0] != 1 and metrics_targets[task][1] != 0 and p_exist and s_exist:
        #             task_mask[i] = 1

        #     elif task == "dis":
        #         if metrics_targets[task].size > 0 and p_exist and metrics_targets[task][0] > 0 and metrics_targets[task][0] < 500:
        #             task_mask[i] = 1

        #     elif task == "emg":
        #         if metrics_targets[task].size > 0 and p_exist and metrics_targets[task][0] >= 0 and metrics_targets[task][0] < 9:
        #             task_mask[i] = 1

        # metrics_targets["task_mask"] = task_mask

        meta_data_json = json.dumps(meta_data)
        # print(inputs,loss_targets,metrics_targets,meta_data_json)
        return inputs, loss_targets, metrics_targets,meta_data_json
