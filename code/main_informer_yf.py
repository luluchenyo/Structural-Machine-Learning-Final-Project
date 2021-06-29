import argparse
import os
import datetime
import tensorflow as tf
import numpy as np
from exp.exp_informer_yf import ExpInformer
import shutil
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import os
import zipfile
def zip_dir(path):
    zf = zipfile.ZipFile('{}.zip'.format(path), 'w', zipfile.ZIP_DEFLATED)

    for root, dirs, files in os.walk(path):
        for file_name in files:
            zf.write(os.path.join(root, file_name))

path = os.path.join('data', 'yfinance', 'tfdata')
if not os.path.isfile('{}.zip'.format(path)):
    zip_dir(path)
# use_gpu = True; server = True; simon = False; test = False
use_gpu = False; server = True; simon = False; test = False
# use_gpu = False; server = False; simon = False; test = False

# use_gpu = True; simon = True
# use_gpu = False; simon = True

# use_gpu = False

# exp_multifeature = [True, False]
exp_start_idx = 12
n_exp = 2
exp_multifeature = [False]
exp_time_embedding = [False]
exp_target_interval = ['1m', '2m']
exp_train_epochs = [1, 4, 64]

models = {}
models['Informer'] = ExpInformer
if use_gpu:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    physical_devices = tf.config.list_physical_devices('CPU')

def args_generator(multifeature, time_embedding, target_interval='2m', train_epochs=1, use_gpu=use_gpu, simon=simon,
                   server=server):
    parser = argparse.ArgumentParser()
    if multifeature:
        parser.add_argument('--enc_in', type=int, default=120, help='encoder input size')
        parser.add_argument('--n_heads', type=int, default=32, help='num of heads')
        parser.add_argument('--factor', type=int, default=8, help='prob sparse factor')
        parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    else:
        parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')
        parser.add_argument('--n_heads', type=int, default=6, help='num of heads')
        parser.add_argument('--factor', type=int, default=4, help='prob sparse factor')
        parser.add_argument('--d_model', type=int, default=64, help='dimension of model')

    # parser.add_argument('--model', type=str, default='informer', help='model of the experiment')

    parser.add_argument('--dec_in', type=int, default=3, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=3, help='output size')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')

    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention [prob, full]')
    parser.add_argument('--embed', type=str, default='fixed', help='embedding type [fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

    parser.add_argument('--itr', type=int, default=2, help='each params run iteration')

    parser.add_argument('--patience', type=int, default=8, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    parser.add_argument('--run_eagerly', type=bool, default=True)

    parser.add_argument('--multifeature', type=bool, default=multifeature)
    parser.add_argument('--time_embedding', type=bool, default=time_embedding, help='if add time stamp')
    parser.add_argument('--train_epochs', type=int, default=train_epochs, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='input data batch size')

    parser.add_argument('--use_gpu', type=bool, default=use_gpu, help='use gpu')
    fn = lambda path: os.path.abspath(path)
    # ('lu', '202102rs', 'luluchen', '20210514code', 'Informer', path)
    # get data

    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    if server:
        parser.add_argument('--root_path', type=str, default=fn(
            os.path.join('lu', '202102rs', 'luluchen', '20210514code', 'Informer', 'data', 'yfinance')),
                            help='root path of the data file')
        parser.add_argument('--checkpoint_path', type=str, default=fn(
            os.path.join('lu', '202102rs', 'luluchen', '20210514code', 'Informer', 'checkpoint')))
    else:
        parser.add_argument('--root_path', type=str, default=os.path.join('data', 'yfinance'),
                            help='root path of the data file')
        parser.add_argument('--checkpoint_path', type=str, default='checkpoint')

    parser.add_argument('--symbol_list', type=list,
                        default=['NOK', '^IXIC', '^DJI', '^GSPC', '^N225', '^FTSE', '^CMC200',
                                 'BTC-USD', '^VIX', '^TNX', 'EURUSD=X', 'SI=F', 'JPY=X', 'GBPUSD=X',
                                 'GC=F', 'CL=F', 'RTY=F', 'NQ=F', 'YM=F', 'ES=F'], help='the features in space')
    parser.add_argument('--target_interval', type=str, default=target_interval)
    parser.add_argument('--target_price', type=str, default='Open')
    parser.add_argument('--target_symbol', type=str, default='^GSPC')
    parser.add_argument('--window_len', type=int, default=1200,
                        help='the pretraining and online training period length')
    parser.add_argument('--test_ratio', type=int, default=0.25,
                        help='the ratio of data used for online training in a window')
    parser.add_argument('--seq_len', type=int, default=64, help='enc input series length')
    parser.add_argument('--dec_len', type=int, default=32, help='dec input series length')
    parser.add_argument('--target_len', type=int, default=24, help='predict series length')

    return parser.parse_args()

# physical_devices = tf.config.list_physical_devices('CPU')
# get data
datasets = {}
targets = {}
for multifeature in exp_multifeature:
    for time_embedding in exp_time_embedding:
        for target_interval in exp_target_interval:
            exp_name = 'Informer'
            if multifeature:
                exp_name += '_space'
            if time_embedding:
                exp_name += '_time'
            exp_name += ('_' + target_interval)
            args = args_generator(multifeature=multifeature, time_embedding=time_embedding,
                                  target_interval=target_interval, use_gpu=use_gpu, simon=simon,
                                  server=server)
            # path = os.path.join(args.root_path, 'tfdata', datetime.date.today().strftime('%Y%m%d'), exp_name)
            # target_path = os.path.join(args.root_path, 'tfdata', datetime.date.today().strftime('%Y%m%d'), 'target')

            tfdata_pathh = os.path.join(args.root_path, 'tfdata')
            dictt = {}
            for dir in os.listdir(tfdata_pathh):
                if dir != 'desktop.ini':
                    if ' ' in dir:
                        dd = int(dir[:dir.find(' ')])
                    else:
                        dd = int(dir)
                    if 30000000>dd>20000000:
                        dictt[dir] = dd
            max_key = max(dictt, key=dictt.get)
            tfdata_path_ori = os.path.join(tfdata_pathh, max_key)

            if server:
                tfdata_path = os.path.join(tfdata_pathh, str(dictt[max_key] - 10000000))
            else:
                tfdata_path = os.path.join(tfdata_pathh, str(dictt[max_key] + 10000000))
            if not os.path.isdir(tfdata_path):
                shutil.copytree(tfdata_path_ori, tfdata_path)
                print('copy data')
            path = os.path.join(tfdata_path, exp_name)

            target_path = os.path.join(tfdata_path, 'target_' + target_interval)
            # print(path)
            # fsdf
            dirss = []
            for dir in os.listdir(path):
                if dir != 'desktop.ini':
                    dirss.append(int(dir))
            # print(np.sort(np.array(dirss)))
            # fsdf
            windows = []
            if test:
                dirss = range(2)
            else:
                dirss = np.sort(np.array(dirss))
            for dir in dirss:
                # print(os.path.join(path, str(dir), 'pretrain'))
                # fdsf
                spec = (tf.TensorSpec((None, args.seq_len, args.enc_in), dtype=tf.float64),
                        tf.TensorSpec((None, args.dec_len, 3), dtype=tf.float64),
                        tf.TensorSpec((None, args.seq_len, 5), dtype=tf.float64),
                        tf.TensorSpec((None, args.dec_len, 5), dtype=tf.float64)), tf.TensorSpec(
                    (None, args.target_len, 3), dtype=tf.float64)
                pretrain = tf.data.experimental.load(os.path.join(path, str(dir), 'pretrain'), element_spec=spec).as_numpy_iterator()
                val = tf.data.experimental.load(os.path.join(path, str(dir), 'val'), element_spec=spec).as_numpy_iterator()
                online = tf.data.experimental.load(os.path.join(path, str(dir), 'online'), element_spec=spec).as_numpy_iterator()
                windows.append((pretrain, val, online))
            metric_target = tf.data.experimental.load(target_path,
                                                      element_spec=tf.TensorSpec(shape=(None, 3), dtype=tf.float64))
            for batch in metric_target:
                metric_target = batch
            metric_target = tf.cast(metric_target, dtype=tf.float32)

            datasets[exp_name] = windows
            targets[target_interval] = metric_target


def trader(output, metric_target):
    policy = tf.argmax(output, axis=1)
    metric_pred = tf.one_hot(policy, depth=3, dtype=tf.float32)
    accu_return = tf.cumsum(tf.reduce_sum(tf.math.multiply(metric_pred, metric_target), axis=1))
    return policy - 1, accu_return


for target_interval in exp_target_interval:
    locals()['traders ({target_interval})'] = {}

features = {}


def feature_PCA(attn_f):
    feature = np.array(attn_f)
    feature = feature.reshape([len(feature), -1])
    model = PCA()
    return model.fit_transform(feature)

if test:
    exp_start_idx *=100
if server:
    exp_start_idx *=50
exp_numbers = range(exp_start_idx, exp_start_idx+n_exp)

# if use_gpu:
#     physical_devices = tf.config.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

for exp_number in exp_numbers:
    for model_ in list(models.keys()):
        for multifeature in exp_multifeature:
            for time_embedding in exp_time_embedding:
                for target_interval in exp_target_interval:
                    metric_target = targets[target_interval]
                    always_buy = tf.concat([[[0., 0., 2.]] for i in range(len(metric_target))], axis=0)
                    locals()['traders ({target_interval})']['always_buy'] = trader(always_buy, metric_target)
                    for train_epochs in exp_train_epochs:
                        exp_name = model_
                        if multifeature:
                            exp_name += '_space'
                        if time_embedding:
                            exp_name += '_time'
                        exp_name += ('_' + target_interval)
                        args = args_generator(multifeature=multifeature, time_embedding=time_embedding,
                                              target_interval=target_interval, use_gpu=use_gpu, simon=simon,
                                              server=server, train_epochs=train_epochs)
                        model = models[model_](args, exp_number=exp_number, exp_name=exp_name,
                                               windows=datasets[exp_name])
                        model.train()
                        locals()['traders ({target_interval})'][exp_name + '_ep' + str(train_epochs)] = trader(model.pred, metric_target)
                        features[exp_name + '_ep' + str(train_epochs)] = feature_PCA(model.attn_f)
    plt.close('all')


    # accumulated profit
    for target_interval in exp_target_interval:
        traders = locals()['traders ({target_interval})']
        col = list(mcolors.TABLEAU_COLORS.keys())
        fig, ax = plt.subplots(1)
        for i in range(len(traders)):
            key = list(traders.keys())[i]
            ax.plot(traders[key][1], color=col[i], label=key)
        plt.legend(frameon=True)
        plt.title('accumulated_profit')
        plt.savefig(os.path.abspath(os.path.join(model.fig_path, 'accumulated_profit')))
        # action
        fig, axs = plt.subplots(len(traders), constrained_layout=True)
        for i in range(len(traders)):
            key = list(traders.keys())[i]
            axs[i].set_title(key)
            y = traders[key][0]
            color = ['red' if item == 1 else 'green' if item == -1 else 'black' for item in y]
            axs[i].scatter(range(len(y)), y, c=color, s=[0.05 for i in range(len(y))])
            axs[i].set_ylim(-2, 2)
        fig.suptitle('action', fontsize=16)
        plt.savefig(os.path.abspath(os.path.join(model.fig_path, 'action_t')))


    # feature

    def k_means(X, axs):
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=3,
                    init='random',
                    n_init=10,
                    max_iter=300,
                    tol=1e-04,
                    random_state=0)

        y_km = km.fit_predict(X)
        # -------------------------------------------------
        axs.scatter(X[y_km == 0, 0],
                    X[y_km == 0, 1],
                    s=20, c='lightgreen',
                    marker='s', edgecolor='black',
                    label='Cluster 1')
        axs.scatter(X[y_km == 1, 0],
                    X[y_km == 1, 1],
                    s=20, c='orange',
                    marker='o', edgecolor='black',
                    label='Cluster 2')
        axs.scatter(X[y_km == 2, 0],
                    X[y_km == 2, 1],
                    s=20, c='lightblue',
                    marker='v', edgecolor='black',
                    label='Cluster 3')
        axs.scatter(km.cluster_centers_[:, 0],
                    km.cluster_centers_[:, 1],
                    s=150, marker='*',
                    c='red', edgecolor='black',
                    label='Centroids')
        axs.legend(scatterpoints=1)
        axs.grid()

    h = len(features)//2; v = len(features) //h
    fig, axs = plt.subplots(v, h, constrained_layout=False)
    i = 0
    for k in range(v):
        for j in range(h):
            try:
                key = list(features.keys())[i]
                axs[k, j].set_title(key)
                X = features[key]

                k_means(X, axs[k, j])
                i+=1
            except:
                break
    fig.suptitle('extrated_features', fontsize=16)

    fig.tight_layout()
    plt.savefig(os.path.abspath(os.path.join(model.fig_path, 'extrated_features')))