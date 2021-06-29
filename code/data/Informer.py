import os
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf

import datetime as dt

ds = lambda data: tf.data.Dataset.from_tensor_slices(data)
zp = lambda data: tf.data.Dataset.zip(data)

warnings.filterwarnings('ignore')

class yfdata():
    def __init__(self, args):
        self.args = args
        self.test_len = int(args.test_ratio * args.window_len)
        if args.target_price == 'Open':
            self.price_col = 0
            if args.multifeature:
                self.price_col += 1
        else:
            self.price_col = 3
            if args.multifeature:
                self.price_col += 1
        if self.args.multifeature:
            self.z = self.load_multi_csv_to_np_z()

            self.n_features_t = self.z.shape[1]
            self.n_features = self.n_features_t -1
        else:
            self.z = self.load_csv_to_np_z()
            self.n_features = self.z.shape[1]

    def __call__(self):

        if self.args.time_embedding:
            time_stamp = self.get_time_stamp(self.z)
            X_y_window_dataset = self.single_feature_to_X_stamp_y_window(self.z, time_stamp)
        else:
            enc, dec, enc_t, dec_t, target = self.feature_to_X_y_window()
        windows = self.X_y_window_to_training_window(enc, dec, enc_t, dec_t, target)
        len_ = len(self.metrics_target_np)

        return tf.data.Dataset.from_tensor_slices(self.metrics_target_np).batch(len_), windows
    def window_processer(self, nparray, sequence_length, stride):
        data = tf.keras.preprocessing.timeseries_dataset_from_array(
            nparray, targets=None, sequence_length=sequence_length, sequence_stride=stride).unbatch()
        return tf.constant(list(data.as_numpy_iterator()))
    def load_multi_csv_to_np_z(self):
        space = {}
        for symbol in self.args.symbol_list:
            path = os.path.join(self.args.root_path, (symbol + self.args.target_interval)) + '.csv'
            try:
                df = pd.read_csv(path, engine='python')
            except:
                print(path)
                fsdf
            def fn(x):
                try:
                    out = dt.datetime.strptime(x[:-3]+x[-2:], '%Y-%m-%d %H:%M:%S%z')
                except:
                    # try:
                    #     out = dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
                    # except:
                    print(path, '!!!!')
                    return None
                return out
            # if self.args.use_gpu:
            #     df.iloc[:, 0] = df.iloc[:, 0].apply(fn)
            # else:
            df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z'))
            df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x.value)
            price_np = np.asarray(df.iloc[:, 1:-1])
            vol_np = np.asarray(df.iloc[1:, -1])[:, np.newaxis]
            t_np = np.asarray(df.iloc[1:, 0])[:, np.newaxis]
            price_np = price_np[1:, :] - price_np[:-1, :]
            data_np = np.concatenate((t_np, price_np, vol_np), axis=1)

            df = pd.DataFrame(data_np)

            df = df[~(df.iloc[:, 0]==None)]
            df.set_index(df.iloc[:, 0], inplace=True)
            space[symbol] = df
            if symbol == self.args.target_symbol:
                target_symbol_np = data_np
        self.space_df = pd.concat(space, axis=1, join='outer', sort=True)
        idx = self.space_df.index
        for symbol in self.args.symbol_list:
            self.space_df[symbol].iloc[:, 0] = idx
        # self.space_df.to_csv('test.csv')


        return target_symbol_np
    def load_csv_to_np_z(self):
        path = os.path.join(self.args.root_path, (self.args.target_symbol + self.args.target_interval))+'.csv'
        df = pd.read_csv(os.path.join(path))
        price_np = np.asarray(df.iloc[:, 1:-1])
        vol_np = np.asarray(df.iloc[1:, -1])[:, np.newaxis]
        price_np = price_np[1:, :] - price_np[:-1, :]
        data_np = np.concatenate((price_np, vol_np), axis=1)

        self.n_features = data_np.shape[1]
        return data_np
    def enc_dec_target_split(self, windows):
        n_windows = len(windows)
        if self.args.multifeature:
            enc = tf.slice(windows, [0, 0, 0], [n_windows, self.args.seq_len, self.n_features_t])
        else:
            enc = tf.slice(windows, [0, 0, 0], [n_windows, self.args.seq_len, self.n_features])
        decc = tf.slice(windows, [0, self.args.seq_len + self.args.target_len - self.args.dec_len, self.price_col],
                       [n_windows, self.args.dec_len - self.args.target_len, 1])
        dec = []
        for d in decc:
            dec.append([[1, 0, 0] if data<0 else [0, 0, 1] if data>0 else [0, 1, 0] for data in d])
        dec = tf.constant(dec, dtype=tf.float64)
        dec = tf.pad(dec, [[0, 0], [0, self.args.target_len], [0, 0]])
        target = tf.slice(windows, [0, self.args.seq_len, self.price_col], [n_windows, self.args.target_len, 1])
        target = tf.concat((tf.multiply(target, -1), tf.zeros([n_windows, self.args.target_len, 1], dtype=tf.float64), tf.multiply(target, 1)), axis=2)
        # enc, dec, target = enc[tf.newaxis, :], dec[tf.newaxis, :], target[tf.newaxis, :]
        shape = list(np.asarray(enc).shape)
        shape[2] = 5
        enc_t = tf.zeros(shape, dtype=tf.float64)

        shape = list(np.asarray(dec).shape)
        shape[2] = 5
        dec_t = tf.zeros(shape, dtype=tf.float64)
        return enc, dec, enc_t, dec_t, target

    def feature_to_X_y_window(self):
        size = self.args.seq_len+self.args.target_len
        data = self.window_processer(nparray=self.z, sequence_length=size, stride=1)
        return self.enc_dec_target_split(data)

    def scaler_X(self, data, window_max, data_len, enc_in):
        return tf.math.divide(data, tf.tile(window_max, [1, data_len, enc_in, 1]))
    def scaler_y(self, data, window_max, data_len, enc_in):
        return tf.math.divide(data, tf.tile(window_max[:, :, :, self.price_col:self.price_col+1],
                                                                [1, data_len, enc_in, 3]))

    def X_y_window_to_training_window(self, enc, dec, enc_t, dec_t, target):
        fn = lambda data: self.window_processer(data, sequence_length=self.args.window_len, stride=self.test_len)
        [enc, dec, enc_t, dec_t, target] = [fn(data) for data in [enc, dec, enc_t, dec_t, target]]
        self.metrics_target_np = []
        online_fn = lambda data: data[:, -self.test_len:]
        metrics_target_np = online_fn(target)
        for window in metrics_target_np:
            for data in window:
                self.metrics_target_np.append([data[-1, :]])

        self.metrics_target_np = tf.concat(self.metrics_target_np, axis=0)

        if self.args.multifeature:
            enc_ = []
            for window in enc:
                window_ = []
                for data in window:
                    time = data[-1, 0]
                    dfs = [data[:, 1:]]
                    for feature in self.args.symbol_list:
                        if feature == self.args.target_symbol:
                            pass
                        else:
                            df = self.space_df[feature]
                            idx = int(np.where(df.iloc[:, 0]==time)[0]) + 1
                            df = df.iloc[:idx, 1:]
                            df.dropna(inplace=True)
                            feature_np = np.array(df)
                            if len(df) < self.args.seq_len:
                                feature_np = np.concatenate((np.zeros([self.args.seq_len-len(df), self.n_features]), feature_np), axis=0)

                            feature_np = feature_np[-self.args.seq_len:, :]

                            dfs.append(feature_np)
                    window_.append(tf.concat(dfs, axis=1)[tf.newaxis, :, :])
                enc_.append(tf.concat(window_, axis=0)[tf.newaxis, :, :, :])
            enc = tf.concat(enc_, axis=0)
            del enc_, window_

        val_len = int(self.test_len / 2)
        pre_len = self.args.window_len - self.test_len - val_len
        pre_fn = lambda data: data[:, :pre_len]
        val_fn = lambda data: data[:, pre_len: -self.test_len]
        pretrain = [pre_fn(data) for data in [enc, dec, enc_t, dec_t, target]]
        val = [val_fn(data) for data in [enc, dec, enc_t, dec_t, target]]
        online = [online_fn(data) for data in [enc, dec, enc_t, dec_t, target]]

        feature_max = tf.math.reduce_max(tf.abs(tf.reshape(pretrain[0], [len(pretrain[0]), -1, self.args.enc_in])), axis=1)

        feature_max = feature_max[:, tf.newaxis, tf.newaxis, :]

        pretrain[0] = self.scaler_X(pretrain[0], feature_max, pre_len, self.args.seq_len)
        val[0] = self.scaler_X(val[0], feature_max, val_len, self.args.seq_len)
        online[0] = self.scaler_X(online[0], feature_max, self.test_len, self.args.seq_len)
        pretrain[-1] = self.scaler_y(pretrain[-1], feature_max, pre_len, self.args.target_len)
        val[-1] = self.scaler_y(val[-1], feature_max, val_len, self.args.target_len)
        online[-1] = self.scaler_y(online[-1], feature_max, self.test_len, self.args.target_len)

        windows = []
        for i in range(len(pretrain[0])):
            windows.append((zp((zp((tuple([ds(pretrain[j][i]).batch(self.args.batch_size) for j in range(len(pretrain)-1)]),
                              ds(pretrain[-1][i]).batch(self.args.batch_size))))), \
                            zp((zp((tuple(
                                [ds(val[j][i]).batch(len(val)) for j in range(len(pretrain) - 1)]),
                                ds(val[-1][i]).batch(len(val)))))), \
                           zp((zp((tuple(
                               [ds(online[j][i]).batch(1) for j in range(len(pretrain)-1)]),
                               ds(online[-1][i]).batch(1)))))))
        return windows



