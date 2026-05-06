import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_provider.m4 import M4Dataset, M4Meta
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe, format_timedelta, convert_time_slot_to_str
import warnings
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
            

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 seasonal_patterns=None, scale=True, drop_short=False):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None, data_path='ETTh1.csv',
                 scale=False, inverse=False, seasonal_patterns='Yearly', drop_short=False):
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class Dataset_TSF(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=False):
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len
        print(self.seq_len, self.label_len, self.pred_len)
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.drop_short = drop_short
        self.timeseries = self.__read_data__()


    def __read_data__(self):
        df, _, _, _, _ = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        if self.drop_short:
            timeseries = [ts for ts in timeseries if ts.shape[0] > self.context_len]
        self.tot_len = 0
        self.len_seq = []
        self.seq_id = []
        for i in range(len(timeseries)):
            res_len = max(self.pred_len + self.seq_len - timeseries[i].shape[0], 0)
            pad_zeros = np.zeros(res_len)
            timeseries[i] = np.hstack([pad_zeros, timeseries[i]])

            _len = timeseries[i].shape[0]
            train_len = _len-self.pred_len
            border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]
            border2s = [train_len - self.pred_len,  train_len,                                _len]
            
            curr_len = border2s[self.set_type] - max(border1s[self.set_type], 0) - self.pred_len - self.seq_len + 1
            curr_len = max(0, curr_len)
            
            self.len_seq.append(np.zeros(curr_len) + self.tot_len)
            self.seq_id.append(np.zeros(curr_len) + i)
            self.tot_len += curr_len
            
        self.len_seq = np.hstack(self.len_seq)
        self.seq_id = np.hstack(self.seq_id)

        return timeseries

    def __getitem__(self, index):
        len_seq = self.len_seq[index]
        seq_id = int(self.seq_id[index])
        index = index - int(len_seq)

        _len = self.timeseries[seq_id].shape[0]
        train_len = _len - self.pred_len
        border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]

        s_begin = index + border1s[self.set_type]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        data_x = self.timeseries[seq_id][s_begin:s_end]
        data_y = self.timeseries[seq_id][r_begin:r_end]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)

        return data_x, data_y, data_x, data_y

    def __len__(self):
        return self.tot_len

class Dataset_TSF_ICL(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=True):
        
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len

        self.root_path = root_path
        self.data_path = data_path
        self.timeseries = self.__read_data__()

    def __read_data__(self):
        df, _, _, _, _ = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        timeseries = [ts for ts in timeseries if ts.shape[0] > self.context_len]
        return timeseries

    # we uniformly adopting the first time points of the time series as the corresponding prompt.
    def __getitem__(self, index):        
        data_x1 = self.timeseries[index][:2*self.token_len]
        data_x2 = self.timeseries[index][-2*self.token_len:-1*self.token_len]
        data_x = np.concatenate((data_x1, data_x2))
        data_y = self.timeseries[index][-1*self.token_len:]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)
        return data_x, data_y, data_x, data_y

    def __len__(self):
        return len(self.timeseries)



class Dataset_FourSquare(Dataset):
    '''
    range: Apr 2012 - Feb 2013
    '''
    def __init__(self, root_path, data_path=None, flag='train', size=None, interval=None,
                 scale=False, city='NYC', **kwargs):
        self.preprocessed_filename = f"foursquare_{city}_{'scaled' if scale else 'unscaled'}_size_{size[0]}_{size[1]}_{size[2]}.npz"
        self.preprocessed_prompt_filename_x = f"foursquare_{city}_{size[0]}_{size[1]}_{size[2]}_x.pt"
        self.preprocessed_prompt_filename_y = f"foursquare_{city}_{size[0]}_{size[1]}_{size[2]}_y.pt"
        self.seq_len = size[0] if size and len(size) > 0 else 40
        self.label_len = size[1] if size and len(size) > 1 else 30
        self.pred_len = size[2] if size and len(size) > 2 else 10
        self.interval = interval if interval is not None else (self.seq_len + self.pred_len)
        self.token_len = self.seq_len - self.label_len  # 10
        self.token_num = self.seq_len // self.token_len  # 4
        self.flag = flag
        self.scale = scale

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        assert city in ['NYC', 'TKY']
        self.city = city
        self.preprocessed_path = os.path.join(root_path, f"{self.preprocessed_filename}")
        self.embed_prompt_path_x = os.path.join(root_path, f"{self.preprocessed_prompt_filename_x}")
        self.embed_prompt_path_y = os.path.join(root_path, f"{self.preprocessed_prompt_filename_y}")
        print(f"Loading preprocessed data from {self.preprocessed_path}")

        self.__read_data__()
        self.__split_data__()  # Perform the split
        print(f"Loaded {self.num_samples} sequences from {self.flag} set.")

    def __read_data__(self):
        """
        Loads data from the preprocessed .npz file.
        """
        data = np.load(self.preprocessed_path, allow_pickle=True)
        self.input_seq_feature = data['input_seq_feature']  # [num_samples, seq_len, 8]
        self.predict_seq_feature = data['predict_seq_feature']  # [num_samples, pred_len, 5]
        self.predict_seq_label = data['predict_seq_label']  # [num_samples, pred_len]
        self.POI_categories = data['POI_categories']  # [num_samples, pred_len] as objects (strings)
        self.timestamps = data['timestamps']  # [num_samples, 2] as objects (start and end timestamps)
        self.num_samples = len(self.input_seq_feature)

        df = pd.read_csv(os.path.join(self.root_path, f'foursquare_{self.city}.csv'))
        self.num_classes = len(np.unique(df['Venue ID']))
        self.num_users = len(np.unique(df['User ID']))
        self.embed_prompts_x = torch.load(self.embed_prompt_path_x)
        self.embed_prompts_y = torch.load(self.embed_prompt_path_y)

    def __split_data__(self):
        """
        Split the data into train, validation, and test sets with fixed seed.
        """
        np.random.seed(42)  # Fixed seed for reproducibility
        indices = np.arange(self.num_samples)
        #np.random.shuffle(indices)

        train_size = int(0.7 * self.num_samples)
        val_size = int(0.2 * self.num_samples)
        test_size = self.num_samples - train_size - val_size


        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]


        if self.flag == 'train':
            self.data_indices = train_indices
        elif self.flag == 'val':
            self.data_indices = val_indices
        elif self.flag == 'test':
            self.data_indices = test_indices

        self.num_samples = len(self.data_indices)

    def get_num_class(self):
        return self.num_classes

    def get_num_users(self):
        return self.num_users

    def __getitem__(self, index):
        """
        Retrieves the data sample at the specified index.

        Returns:
            tuple: (
                input_seq_feature (torch.Tensor),      # [seq_len, 8]
                predict_seq_feature (torch.Tensor),    # [pred_len, 5]
                predict_seq_label (torch.Tensor),      # [pred_len]
                prompt_embedding_x (torch.Tensor),     # [prompt_embedding_size]
                prompt_embedding_y (torch.Tensor]     # [prompt_embedding_size]
            )
        """
        real_index = self.data_indices[index]
        input_feat = self.input_seq_feature[real_index]  # [seq_len, 7]
        output_feat = self.predict_seq_feature[real_index]  # [pred_len, 4]
        output_label = self.predict_seq_label[real_index]  # [pred_len]

        return (
            torch.tensor(input_feat, dtype=torch.float32),  # [seq_len, 7]
            torch.tensor(output_feat, dtype=torch.float32),  # [pred_len, 4]
            torch.tensor(output_label, dtype=torch.int),  # [pred_len]
            self.embed_prompts_x[real_index],  # [prompt_embedding_size]
            self.embed_prompts_y[real_index],
        )

    def __len__(self):
        """
        Calculate the total number of valid samples.
        """
        return self.num_samples

    def inverse_transform(self, data):
        """
        Revert scaled data back to original scale.
        """
        return self.scaler.inverse_transform(data)

class Dataset_YJ(Dataset):
    def __init__(self, root_path, data_path=None, flag='train', size=None, llm_ckp_dir='meta-llama/Llama-3.2-1B',
                 scale=False, city=None, **kwargs):
        self.preprocessed_filename = f"yj_{city}_size_336_288_48_{flag}.npz"
        model_abbrev = llm_ckp_dir.split('/')[-1]
        # self.preprocessed_prompt_filename_x = f"yj_{city}_{size[0]}_{size[1]}_{size[2]}_1B_x.pt"
        # self.preprocessed_prompt_filename_y = f"yj_{city}_{size[0]}_{size[1]}_{size[2]}_{flag}_1B_y.pt"
        self.preprocessed_prompt_filename_x = f"yj_{city}_{size[0]}_{size[1]}_{size[2]}_{model_abbrev}_x.pt"
        self.preprocessed_prompt_filename_y = f"yj_{city}_{size[0]}_{size[1]}_{size[2]}_{flag}_{model_abbrev}_y.pt"
        self.seq_len = size[0] if size and len(size) > 0 else 40
        self.label_len = size[1] if size and len(size) > 1 else 30
        self.pred_len = size[2] if size and len(size) > 2 else 10
        self.token_len = self.seq_len - self.label_len  # 48
        self.token_num = self.seq_len // self.token_len  # 7
        self.flag = flag
        self.scale = scale

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        assert city in ['A', 'B', 'C', 'D', 'BOS']
        self.city = city
        self.preprocessed_path = os.path.join(root_path, f"{self.preprocessed_filename}")
        self.embed_prompt_path_x = os.path.join(root_path, f"{self.preprocessed_prompt_filename_x}")
        self.embed_prompt_path_y = os.path.join(root_path, f"{self.preprocessed_prompt_filename_y}")
        print(f"Loading preprocessed data from {self.preprocessed_path}")

        self.__read_data__()
        # self.__split_data__()  # Perform the split
        print(f"Loaded {self.num_samples} sequences from {self.flag} set.")

    def __read_data__(self):
        """
        Loads data from the preprocessed .npz file.
        """
        data = np.load(self.preprocessed_path, allow_pickle=True)
        self.input_seq_feature = data['input_seq_feature']  # [num_samples, seq_len, 7]
        self.predict_seq_feature = data['predict_seq_feature']  # [num_samples, pred_len, 4]
        self.predict_seq_label = data['predict_seq_label']  # [num_samples, pred_len]
        self.num_samples = len(self.input_seq_feature)

        self.num_classes = 40001
        # print(self.embed_prompt_path_x)
        self.embed_prompts_x = torch.load(self.embed_prompt_path_x)
        self.embed_prompts_y = torch.load(self.embed_prompt_path_y)
        # print("embed_prompts_x:", tuple(self.embed_prompts_x.shape))
        # print("embed_prompts_y:", tuple(self.embed_prompts_y.shape))
        # print("Prompt dim H =", self.embed_prompts_y.shape[-1])
        # newly added for RHYTHM high-freq planning
        if 'predict_seq_feature_full_7d' in data.files:
            self.predict_seq_feature_full_7d = data['predict_seq_feature_full_7d']  # [N, 336, 7]
            print("Loaded predict_seq_feature_full_7d for RHYTHM high-freq planning.")
        else:
            self.predict_seq_feature_full_7d = None

        uids = self.input_seq_feature[:, 0, 0].astype(np.int64)
        start_days = self.input_seq_feature[:, 0, 3].astype(np.int64)
        self._idx_map = {(int(u), int(d)): i for i, (u, d) in enumerate(zip(uids, start_days))}

    def get_num_class(self):
        return 40001 # 40000 for missing

    def get_num_users(self):
        return len(np.unique(self.input_seq_feature[:, 0, 0]))

    def __getitem__(self, index):
        """
        Retrieves the data sample at the specified index.
        ['User ID', 'time slot', 'DayOfWeek', 'Day', 'Latitude', 'Longitude', 'Place ID']
        Returns:
            tuple: (
                input_seq_feature (torch.Tensor),      # [seq_len, 7]
                predict_seq_feature (torch.Tensor),    # [pred_len, 4]
                predict_seq_label (torch.Tensor),      # [pred_len]
                prompt_embedding_x (torch.Tensor),     # [token_num, prompt_embedding_size]
                prompt_embedding_y (torch.Tensor]     # [prompt_embedding_size]
            )
        """
        input_feat = self.input_seq_feature[index]  # [seq_len, 7]
        output_feat = self.predict_seq_feature[index]  # [pred_len, 4]
        output_label = self.predict_seq_label[index]  # [pred_len]
        
        uid = int(input_feat[0, 0])
        start_day = int(input_feat[0, 3])
        indices = (uid * 75 + start_day + torch.arange(self.token_num)).long()
        embedded_token_prompt = self.embed_prompts_x[indices]
        embedded_token_prompt = embedded_token_prompt.view(self.token_num, -1)  # [token_num, prompt_embedding_size]
        future_full_7d = None
        if self.predict_seq_feature_full_7d is not None:
            future_full_7d = torch.tensor(self.predict_seq_feature_full_7d[index], dtype=torch.float32)  # [336,7]
        else:
            # If the old npz does not have this field, return an empty tensor to prevent dataloader crash
            future_full_7d = torch.empty(0)

        return (
            torch.tensor(input_feat, dtype=torch.float32),  # [seq_len, 7]
            torch.tensor(output_feat, dtype=torch.float32),  # [pred_len, 4]
            torch.tensor(output_label, dtype=torch.int),  # [pred_len]
            embedded_token_prompt,  # [token_num, prompt_embedding_size] 
            self.embed_prompts_y[index],
            future_full_7d  # [336,7] or empty tensor
        )
    def get_x_prompt(self, uid_tensor: torch.Tensor, start_day_tensor: torch.Tensor, device=None):
        """
        returns [B, token_num, hidden] x prompt embedding
        """
        if device is None:
            device = uid_tensor.device
        uid = uid_tensor.detach().cpu().long()
        sd = start_day_tensor.detach().cpu().long()

        # indices: (uid*75 + (start_day + 0..6))
        ar = torch.arange(self.token_num, dtype=torch.long).view(1, -1)  # [1,7]
        indices = (uid.view(-1, 1) * 75 + sd.view(-1, 1) + ar).long()    # [B,7]
        x = self.embed_prompts_x[indices].view(uid.shape[0], self.token_num, -1)

        return x.to(device)


    def get_y_prompt(self, uid_tensor: torch.Tensor, start_day_tensor: torch.Tensor, device=None):
        """
        returns [B, hidden] y prompt embedding
        rule: one sample's y prompt corresponds to the prediction on "start_day + token_num(=7)" day.
        To get the y prompt for the k-th rollout (k=0/1/2), we should look for the embed_prompts_y of the sample with start_day+k.

        """
        if device is None:
            device = uid_tensor.device
        uid = uid_tensor.detach().cpu().long().numpy()
        sd = start_day_tensor.detach().cpu().long().numpy()

        idxs = []
        for u, d in zip(uid, sd):
            key = (int(u), int(d))
            if key in self._idx_map:
                idxs.append(self._idx_map[key])
            else:
                idxs.append(0)  #
        # idxs = torch.tensor(idxs, dtype=torch.long, device=device)
        idxs = torch.tensor(idxs, dtype=torch.long)
        y = self.embed_prompts_y[idxs]
        return y.to(device)
    
    def get_y_prompt_horizon_concat(self, uid_tensor, start_day_tensor, horizon=3, device=None):
        """
        Return concatenated daily y-prompts over a horizon.
        Shape: [B, horizon*H]
        """
        if device is None:
            device = uid_tensor.device

        uid_cpu = uid_tensor.detach().cpu().long().numpy()
        sd_cpu = start_day_tensor.detach().cpu().long().numpy()

        idxs_list = []
        for u, d in zip(uid_cpu, sd_cpu):
            ids = []
            for k in range(horizon):
                key = (int(u), int(d + k))
                ids.append(self._idx_map.get(key, 0))
            idxs_list.append(ids)

        idxs = torch.tensor(idxs_list, dtype=torch.long)  # [B, horizon] on CPU
        prompts = self.embed_prompts_y[idxs]              # [B, horizon, H] on CPU
        prompts = prompts.reshape(prompts.size(0), -1)    # [B, horizon*H]
        return prompts.to(device)

    def __len__(self):
        """
        Calculate the total number of valid samples.
        """
        return self.num_samples


class Dataset_Preprocess(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv', scale=True, seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.data_set_type = data_path.split('.')[0]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_stamp)

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date).apply(str)
        self.data_stamp = df_stamp['date'].values
        self.data_stamp = [str(x) for x in self.data_stamp]
        
    # TODO: Prompt
    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.token_len
        start = datetime.datetime.strptime(self.data_stamp[s_begin], "%Y-%m-%d %H:%M:%S")
        if self.data_set_type in ['traffic', 'electricity', 'ETTh1', 'ETTh2']:
            end = (start + datetime.timedelta(hours=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        elif self.data_set_type == 'weather':
            end = (start + datetime.timedelta(minutes=10*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        elif self.data_set_type in ['ETTm1', 'ETTm2']:
            end = (start + datetime.timedelta(minutes=15*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        seq_x_mark = f"This is Time Series from {self.data_stamp[s_begin]} to {end}"
        return seq_x_mark

    def __len__(self):
        return len(self.data_stamp)


    
    
class Dataset_Preprocess_Foursquare(Dataset):
    '''
    range: Apr 2012 - Feb 2013
    '''
    def __init__(self, root_path, flag='train', size=None, interval=None,
                 scale=False, city='NYC', preprocessed_filename = 'foursquare_NYC_unscaled_size_40_35_5.npz'):
        self.seq_len = size[0] if size and len(size) > 0 else 40
        self.label_len = size[1] if size and len(size) > 1 else 30
        self.pred_len = size[2] if size and len(size) > 2 else 10
        self.interval = interval if interval is not None else (self.seq_len + self.pred_len)
        
        self.token_len = self.seq_len - self.label_len # 10
        self.token_num = self.seq_len // self.token_len # 4
        self.flag = flag
        self.scale = scale

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        
        assert city in ['NYC', 'TKY']
        self.city = city
        self.preprocessed_path = os.path.join(root_path, f"dataset/foursquare/{preprocessed_filename}")
        print(f"Loading preprocessed data from {self.preprocessed_path}")

        self.__read_data__()
        

    def __read_data__(self):
        """
        Loads data from the preprocessed .npz file.
        """
        data = np.load(self.preprocessed_path, allow_pickle=True)
        self.input_seq_feature = data['input_seq_feature']             # [num_samples, seq_len, 7]
        self.predict_seq_feature = data['predict_seq_feature']         # [num_samples, pred_len, 4]
        self.predict_seq_label = data['predict_seq_label']             # [num_samples, pred_len]
        self.POI_categories = data['POI_categories']                   # [num_samples, pred_len] as objects (strings)
        self.timestamps = data['timestamps']                           # [num_samples, 2] as objects (start and end timestamps)
        self.num_samples = len(self.input_seq_feature)
        print(f"Loaded {self.num_samples} sequences from preprocessed data.")
    

    def __getitem__(self, index):
        """
        Generate the prompt with enhanced temporal details for prediction.
        """
        # Retrieve the data
        input_feat = self.input_seq_feature[index]          # [seq_len, 7]
        poi_cat = self.POI_categories[index]                # [pred_len]
        prompts = []
        
        for i in range(0, self.token_num):
            start_ts, end_ts = self.timestamps[index][i*self.token_len], self.timestamps[index][self.token_len * (i+1) -1 ]
        
            # Ensure timestamps are strings in desired format
            formatted_start_ts = pd.to_datetime(start_ts).strftime('%Y-%m-%d %H:%M')
            formatted_end_ts = pd.to_datetime(end_ts).strftime('%Y-%m-%d %H:%M')
        
            # Calculate deltas between timestamps in the input sequence
            time_deltas = []
            for j in range(i*self.token_len + 1, (i+1)*self.token_len):
                delta = (
                    pd.to_datetime(self.timestamps[index][j]) - 
                    pd.to_datetime(self.timestamps[index][j - 1])
                )
                time_deltas.append(delta)

            formatted_time_deltas = [format_timedelta(delta) for delta in time_deltas]


            # Initialize an empty list to store formatted records
            record_list = []
            user_id = int(input_feat[0, 0])
            for j in range(i*self.token_len + 1, (i+1)*self.token_len):
                latitude = input_feat[j, 5]
                longitude = input_feat[j, 6]
                venue_id = int(input_feat[j, 7])
                poi_category = poi_cat[j]
                record_time = self.timestamps[index][j]
                formatted_time = pd.to_datetime(record_time).strftime('%Y-%m-%d %H:%M')
                record_list.append(f"({formatted_time}, {venue_id}, {poi_category}, {latitude}, {longitude})")


            # Join all records into a single string
            records_str = ", ".join(record_list)

            # Construct the revised prompt with detailed time delta information
            time_deltas_str = ", ".join(formatted_time_deltas)
            prompt = (
                f"This is the historical trajectory of user {user_id} from {formatted_start_ts} to {formatted_end_ts}. "
                f"The trajectory consists of {self.token_len} records, each formatted as (time, POI ID, POI category, latitude, longitude): {records_str}. "
                f"The time deltas between consecutive records are: {time_deltas_str}."
            )
            prompts.append(prompt)
        # Append the delta between the last input timestamp and the first prediction timestamp
        last_input_ts = pd.to_datetime(self.timestamps[index][self.seq_len - 1])
        first_pred_ts = pd.to_datetime(self.timestamps[index][self.seq_len])
        pred_start_delta = first_pred_ts - last_input_ts
        formatted_pred_start_delta = format_timedelta(pred_start_delta)

        # Extract the prediction timestamps
        pred_timestamps = self.timestamps[index][self.seq_len:self.seq_len + self.pred_len]
        formatted_pred_timestamps = [
            pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M') for ts in pred_timestamps
        ]
        pred_timestamps_str = ", ".join(formatted_pred_timestamps)
        # Number of records to predict

        pred_records = self.pred_len

        prompt1 = (
            f"the time delta between the last record and the next prediction start timestamp is {formatted_pred_start_delta}."
            f"Here are the next {pred_records} timestamps: {pred_timestamps_str}. "
            f"Given each of these timestamps, predict the POI ID where the user will stay."
        )
        
        return prompts, prompt1


    def __len__(self):
        """
        Calculate the total number of valid samples.
        """
        return self.num_samples
    

class Dataset_Preprocess_YJ_Token(Dataset):
    def __init__(self, root_path, size=None,
                 scale=False, city='D'):
        self.seq_len = size[0] if size and len(size) > 0 else 7*48
        self.label_len = size[1] if size and len(size) > 1 else 6*48
        self.pred_len = size[2] if size and len(size) > 2 else 48
        
        self.token_len = self.seq_len - self.label_len # 48
        self.token_num = self.seq_len // self.token_len # 7
        self.scale = scale
        preprocessed_filename = f"yj_{city}_full_sequence.npy"

        self.root_path = root_path
        
        assert city in ['A', 'B', 'C', 'D', 'BOS']
        self.city = city
        self.preprocessed_path = os.path.join(root_path, f"dataset/yj/{preprocessed_filename}")
        print(f"Loading preprocessed data from {self.preprocessed_path}")

        self.__read_data__()
        

    def __read_data__(self):
        """
        Loads data from the preprocessed .npy file.
        """
        self.data = np.load(self.preprocessed_path, allow_pickle=True)
        self.num_uids = self.data.shape[0]
        self.num_days = self.data.shape[1]
    

    def __getitem__(self, index):
        """
        Generate the prompt with enhanced temporal details, key transitions, and stay durations.
        """
        user_id = index // self.num_days
        day_id = index % self.num_days
        day_of_week_id = day_id % 7
        
        # Retrieve the data
        if self.city == 'BOS':
            day_of_week_dict = {0: 'Wednesday', 1: 'Thursday', 2: 'Friday', 3: 'Saturday', 4: 'Sunday', 5: 'Monday', 6: 'Tuesday'}
        else:
            day_of_week_dict = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        
        input_feat = self.data[user_id, day_id]  # [seq_len, 7]
        
        # Process trajectory data
        record_list = []
        user_id_val = int(input_feat[0, 0])
        
        # Track points for transition and stay analysis
        valid_points = []
        
        for j in range(0, self.token_len):
            latitude = input_feat[j, 4]
            if latitude == 999:
                continue
                
            longitude = input_feat[j, 5]
            time_slot = int(input_feat[j, 1])
            time_str = convert_time_slot_to_str(time_slot)
            
            # Add to record list
            record_list.append(f"{time_str}: (X={int(latitude)}, Y={int(longitude)})")
            
            # Store valid point
            valid_points.append({
                'coords': (int(latitude), int(longitude)),
                'time_slot': time_slot,
                'time_str': time_str
            })
        
        # Basic prompt without transitions or stays
        prompt = (
            f"This is the trajectory of user {user_id_val} of day {int(day_id)} which is a {day_of_week_dict[int(day_of_week_id)]}. "
            f"The trajectory consists of {len(record_list)} records, each record of coordinate is as follows: {'; '.join(record_list)}. "
        )
        
        # Only add transitions and stays if we have enough data points
        if len(valid_points) >= 3:
            # Define significant transition threshold (adjust based on coordinate scale)
            TRANSITION_THRESHOLD = 5
            
            # Identify transitions and stays
            transitions = []
            stay_locations = []
            current_cluster = [valid_points[0]]
            
            for i in range(1, len(valid_points)):
                prev_point = valid_points[i-1]
                curr_point = valid_points[i]
                
                # Calculate distance
                distance = ((curr_point['coords'][0] - prev_point['coords'][0])**2 + 
                            (curr_point['coords'][1] - prev_point['coords'][1])**2)**0.5
                
                if distance > TRANSITION_THRESHOLD:
                    # Found a transition
                    
                    # Process previous cluster as a stay
                    if len(current_cluster) >= 2:
                        avg_lat = sum(p['coords'][0] for p in current_cluster) / len(current_cluster)
                        avg_lon = sum(p['coords'][1] for p in current_cluster) / len(current_cluster)
                        duration = (current_cluster[-1]['time_slot'] - current_cluster[0]['time_slot']) / 2  # in hours
                        
                        if duration >= 0.5:  # Only record stays of at least 30 minutes
                            stay_locations.append({
                                'coords': (int(avg_lat), int(avg_lon)),
                                'start': current_cluster[0]['time_str'],
                                'end': current_cluster[-1]['time_str'],
                                'duration': duration
                            })
                    
                    # Record transition
                    transitions.append({
                        'from': prev_point['coords'],
                        'to': curr_point['coords'],
                        'time': curr_point['time_str'],
                        'distance': distance
                    })
                    
                    # Start new cluster
                    current_cluster = [curr_point]
                else:
                    # Continue current cluster
                    current_cluster.append(curr_point)
            
            # Process final cluster
            if len(current_cluster) >= 2:
                avg_lat = sum(p['coords'][0] for p in current_cluster) / len(current_cluster)
                avg_lon = sum(p['coords'][1] for p in current_cluster) / len(current_cluster)
                duration = (current_cluster[-1]['time_slot'] - current_cluster[0]['time_slot']) / 2
                
                if duration >= 0.5:
                    stay_locations.append({
                        'coords': (int(avg_lat), int(avg_lon)),
                        'start': current_cluster[0]['time_str'],
                        'end': current_cluster[-1]['time_str'],
                        'duration': duration
                    })
            
            # Add key transitions to prompt (if any)
            if transitions:
                # Sort by distance (largest first) and take top 3
                transitions.sort(key=lambda x: x['distance'], reverse=True)
                transition_strs = []
                
                for t in transitions[:min(3, len(transitions))]:
                    transition_strs.append(
                        f"At {t['time']}: (X={t['from'][0]}, Y={t['from'][1]}) → (X={t['to'][0]}, Y={t['to'][1]})"
                    )
                
                prompt += f"\n\nKey transitions: {'; '.join(transition_strs)}."
            
            # Add stay locations to prompt (if any)
            if stay_locations:
                # Sort by duration (longest first) and take top 3
                stay_locations.sort(key=lambda x: x['duration'], reverse=True)
                stay_strs = []
                
                for s in stay_locations[:min(3, len(stay_locations))]:
                    stay_strs.append(
                        f"(X={s['coords'][0]}, Y={s['coords'][1]}) from {s['start']} to {s['end']} ({s['duration']:.1f} hours)"
                    )
                
                prompt += f"\n\nMain stay locations: {'; '.join(stay_strs)}."
        
        return prompt


    def __len__(self):
        """
        Calculate the total number of valid samples.
        """
        return self.num_uids * self.num_days
    
class Dataset_Preprocess_YJ(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 scale=False, city='D'):
        self.seq_len = size[0] if size and len(size) > 0 else 7*48
        self.label_len = size[1] if size and len(size) > 1 else 6*48
        self.pred_len = size[2] if size and len(size) > 2 else 48
        
        self.token_len = self.seq_len - self.label_len # 48
        self.token_num = self.seq_len // self.token_len # 7
        self.flag = flag
        self.scale = scale
        preprocessed_filename = f"yj_{city}_size_{self.seq_len}_{self.label_len}_{self.pred_len}_{self.flag}.npz"

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        
        assert city in ['A', 'B', 'C', 'D', 'BOS']
        self.city = city
        self.preprocessed_path = os.path.join(root_path, f"dataset/yj/{preprocessed_filename}")
        print(f"Loading preprocessed data from {self.preprocessed_path}")

        self.__read_data__()
        

    def __read_data__(self):
        """
        Loads data from the preprocessed .npz file.
        """
        data = np.load(self.preprocessed_path, allow_pickle=True)
        self.input_seq_feature = data['input_seq_feature']             # [num_samples, seq_len, 7]
        self.predict_seq_feature = data['predict_seq_feature']         # [num_samples, pred_len, 4]
        self.predict_seq_label = data['predict_seq_label']             # [num_samples, pred_len]
        self.num_samples = len(self.input_seq_feature)
        print(f"Loaded {self.num_samples} sequences from preprocessed data.")
    

    def __getitem__(self, index):
        """
        Generate an enhanced prompt that clearly defines the mobility prediction task.
        """
        # Retrieve the data
        if self.city == 'BOS':
            day_of_week_dict = {0: 'Wednesday', 1: 'Thursday', 2: 'Friday', 3: 'Saturday', 4: 'Sunday', 5: 'Monday', 6: 'Tuesday'}
        else:
            day_of_week_dict = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        
        input_feat = self.input_seq_feature[index]  # [seq_len, 7]
        user_id = int(input_feat[0, 0])
        day = int(self.predict_seq_feature[index][0, 3])
        day_of_week = day_of_week_dict[input_feat[0, 2]]
        
        prompt = (
            "You are a mobility prediction assistant that forecasts human movement patterns in urban environments. "
            "The city is represented as a 200 x 200 grid of cells, where each cell is identified by coordinates (X,Y). "
            "The X coordinate increases from left (0) to right (199), and the Y coordinate increases from top (0) to bottom (199). "
            "\n\n"
            f"TASK: Based on User {user_id}'s historical movement patterns, predict their locations for Day {day} ({day_of_week}). "
            "The predictions should capture expected locations at 30-minute intervals throughout the day (48 time slots). "
            "The model should analyze patterns like frequent locations, typical daily routines, and time-dependent behaviors "
            "to generate accurate predictions of where this user is likely to be throughout the next day."
            "\n\n"
            "The previous days' trajectory data contains information about the user's typical movement patterns, regular visited locations, "
            "transition times, and duration of stays. Key patterns to consider include: home and work locations, morning and evening routines, "
            "lunch-time behaviors, weekend vs. weekday differences, and recurring visit patterns."
        )
        
        return prompt


    def __len__(self):
        """
        Calculate the total number of valid samples.
        """
        return self.num_samples
    
class Dataset_Preprocess_US(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 scale=False, city='D'):
        self.seq_len = size[0] if size and len(size) > 0 else 7*48
        self.label_len = size[1] if size and len(size) > 1 else 6*48
        self.pred_len = size[2] if size and len(size) > 2 else 48
        
        self.token_len = self.seq_len - self.label_len # 48
        self.token_num = self.seq_len // self.token_len # 7
        self.flag = flag
        self.scale = scale
        preprocessed_filename = f"us_{city}_size_{self.seq_len}_{self.label_len}_{self.pred_len}_{self.flag}.npz"

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        
        assert city in ['BOS', 'LA', 'NYC']
        self.city = city
        self.preprocessed_path = os.path.join(root_path, f"dataset/us/{city}/{preprocessed_filename}")
        print(f"Loading preprocessed data from {self.preprocessed_path}")

        self.__read_data__()
        

    def __read_data__(self):
        """
        Loads data from the preprocessed .npz file.
        """
        data = np.load(self.preprocessed_path, allow_pickle=True)
        self.input_seq_feature = data['input_seq_feature']             # [num_samples, seq_len, 7]
        self.predict_seq_feature = data['predict_seq_feature']         # [num_samples, pred_len, 4]
        self.predict_seq_label = data['predict_seq_label']             # [num_samples, pred_len]
        self.num_samples = len(self.input_seq_feature)
        print(f"Loaded {self.num_samples} sequences from preprocessed data.")
    

    def __getitem__(self, index):
        """
        Generate the prompt with enhanced temporal details for prediction.
        """
        # Retrieve the data
        day_of_week_dict = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        input_feat = self.input_seq_feature[index]          # [seq_len, 7]
        user_id = int(input_feat[0, 0])
        day = self.predict_seq_feature[index][0, 3]        

        prompt1 = (
            "You are a helpful assistant that predicts human mobility trajectories in a city. "
            "The target city is divided into Census Block Groups, with each group's location represented by the latitude and longitude of its centroid. "
            "Missing observations are denoted as (999, 999). "
            f"Given the trajectory of user {user_id}, your task is to predict the location records for user {user_id} on the next day, day {int(day)}, which falls on a {day_of_week_dict[input_feat[0, 2]]}."
        )
        
        return prompt1


    def __len__(self):
        """
        Calculate the total number of valid samples.
        """
        return self.num_samples


class Dataset_Preprocess_US_Token(Dataset):
    def __init__(self, root_path, size=None,
                 scale=False, city='D'):
        self.seq_len = size[0] if size and len(size) > 0 else 7*48
        self.label_len = size[1] if size and len(size) > 1 else 6*48
        self.pred_len = size[2] if size and len(size) > 2 else 48
        
        self.token_len = self.seq_len - self.label_len # 48
        self.token_num = self.seq_len // self.token_len # 7
        self.scale = scale
        preprocessed_filename = f"us_{city}_full_sequence.npy"

        self.root_path = root_path
        
        assert city in ['BOS', 'LA', 'NYC']
        self.city = city
        self.preprocessed_path = os.path.join(root_path, f"dataset/us/{city}/{preprocessed_filename}")
        print(f"Loading preprocessed data from {self.preprocessed_path}")

        self.__read_data__()
        

    def __read_data__(self):
        """
        Loads data from the preprocessed .npy file.
        """
        self.data = np.load(self.preprocessed_path, allow_pickle=True)
        self.num_uids = self.data.shape[0]
        self.num_days = self.data.shape[1]
        print(f"Loaded {self.num_uids} users and {self.num_days} days from preprocessed data.")
    

    def __getitem__(self, index):
        """
        Generate the prompt with enhanced temporal details for prediction.
        """
        user_id = index // self.num_days
        day_id = index % self.num_days
        day_of_week_id = (day_id + 3) % 7
        # Retrieve the data
        day_of_week_dict = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        input_feat = self.data[user_id, day_id]          # [seq_len, 7]
        

        # Initialize an empty list to store formatted records
        record_list = []
        user_id = int(input_feat[0, 0])
        for j in range(0, self.token_len):
            latitude = input_feat[j, 4]
            if latitude == 999:
                    continue
            longitude = input_feat[j, 5]
            time_str = convert_time_slot_to_str(input_feat[j, 1])
            record_list.append(f"{time_str}: ({latitude:.4f}, {longitude:.4f})")


        # Join all records into a single string
        records_str = "; ".join(record_list)
        prompt = (
            f"This is the trajectory of user {user_id} of day {int(day_id)} which is a {day_of_week_dict[int(day_of_week_id)]}. "
            f"The trajectory consists of {len(record_list)} records, each record of coordinate is as follows: {records_str}. "
        )
        
        return prompt


    def __len__(self):
        """
        Calculate the total number of valid samples.
        """
        return self.num_uids * self.num_days
    
class Dataset_US(Dataset):
    def __init__(self, root_path='dataset/us/', data_path=None, flag='train', size=None, llm_ckp_dir='meta-llama/Llama-3.2-1B',
                 scale=False, city=None, **kwargs):
        self.preprocessed_filename = f"us_{city}_size_336_288_48_{flag}.npz"
        model_size = llm_ckp_dir[-2:]
        self.preprocessed_prompt_filename_x = f"us_{city}_{size[0]}_{size[1]}_{size[2]}_{model_size}_x.pt"
        self.preprocessed_prompt_filename_y = f"us_{city}_{size[0]}_{size[1]}_{size[2]}_{flag}_{model_size}_y.pt"
        self.seq_len = size[0] if size and len(size) > 0 else 40
        self.label_len = size[1] if size and len(size) > 1 else 30
        self.pred_len = size[2] if size and len(size) > 2 else 10
        self.token_len = self.seq_len - self.label_len  # 48
        self.token_num = self.seq_len // self.token_len  # 7
        self.flag = flag
        self.scale = scale

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        assert city in ['BOS', 'LA', 'NYC']
        self.city = city
        self.preprocessed_path = os.path.join(root_path, f"{city}/{self.preprocessed_filename}")
        self.embed_prompt_path_x = os.path.join(root_path, f"{self.preprocessed_prompt_filename_x}")
        self.embed_prompt_path_y = os.path.join(root_path, f"{self.preprocessed_prompt_filename_y}")
        print(f"Loading preprocessed data from {self.preprocessed_path}")

        self.__read_data__()
        # self.__split_data__()  # Perform the split
        print(f"Loaded {self.num_samples} sequences from {self.flag} set.")

    def __read_data__(self):
        """
        Loads data from the preprocessed .npz file.
        """
        data = np.load(self.preprocessed_path, allow_pickle=True)
        self.input_seq_feature = data['input_seq_feature']  # [num_samples, seq_len, 7]
        self.predict_seq_feature = data['predict_seq_feature']  # [num_samples, pred_len, 4]
        self.predict_seq_label = data['predict_seq_label']  # [num_samples, pred_len]
        self.num_samples = len(self.input_seq_feature)

        self.embed_prompts_x = torch.load(self.embed_prompt_path_x)
        self.embed_prompts_y = torch.load(self.embed_prompt_path_y)

    def get_num_class(self):
        if self.city == 'BOS':
            return 4000
        elif self.city == 'LA':
            return 9000
        elif self.city == 'NYC':
            return 15000

    def get_num_users(self):
        return len(np.unique(self.input_seq_feature[:, 0, 0]))

    def __getitem__(self, index):
        """
        Retrieves the data sample at the specified index.
        ['User ID', 'time slot', 'DayOfWeek', 'Day', 'Latitude', 'Longitude', 'Place ID']
        Returns:
            tuple: (
                input_seq_feature (torch.Tensor),      # [seq_len, 7]
                predict_seq_feature (torch.Tensor),    # [pred_len, 4]
                predict_seq_label (torch.Tensor),      # [pred_len]
                prompt_embedding_x (torch.Tensor),     # [prompt_embedding_size]
                prompt_embedding_y (torch.Tensor]     # [prompt_embedding_size]
            )
        """
        input_feat = self.input_seq_feature[index]  # [seq_len, 7]
        output_feat = self.predict_seq_feature[index]  # [pred_len, 4]
        output_label = self.predict_seq_label[index]  # [pred_len]
        
        uid = int(input_feat[0, 0])
        start_day = int(input_feat[0, 3])
        # print(uid, start_day)
        embedded_token_prompt = []
        for i in range(self.token_num):
            token_index = uid * 59 + start_day + i
            embedded_token_prompt.append(self.embed_prompts_x[token_index])
        embedded_token_prompt = torch.stack(embedded_token_prompt, dim=0) # [token_num, prompt_embedding_size]

        return (
            torch.tensor(input_feat, dtype=torch.float32),  # [seq_len, 7]
            torch.tensor(output_feat, dtype=torch.float32),  # [pred_len, 4]
            torch.tensor(output_label, dtype=torch.int),  # [pred_len]
            embedded_token_prompt,  # [token_num, prompt_embedding_size] 
            self.embed_prompts_y[index],
        )

    def __len__(self):
        """
        Calculate the total number of valid samples.
        """
        return self.num_samples