# import sys
# sys.path.append("..")
from models.Informer.model import Informer
import tensorflow as tf
import os
from tqdm import tqdm

class ExpInformer(Informer):
    def __init__(self, args, exp_number, exp_name, windows):
        super(ExpInformer, self).__init__(args)
        self.fig_path = os.path.join(args.checkpoint_path, str(exp_number))
        self.checkpoint_path = os.path.join(args.checkpoint_path, str(exp_number), exp_name)
        if not os.path.isdir(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.args = args
        self.windows = windows

        # for training loss
        w = tf.reduce_mean(range(1, args.target_len+1))
        self.label_weight = tf.convert_to_tensor([[tf.math.divide(i, w), tf.math.divide(i, w), tf.math.divide(i, w)] for i in range(1, args.target_len+1)], dtype=tf.float32)

        # self.label_weight = tf.constant(self.label_weight, dtype=tf.float32)
        # print(self.label_weight)
        # fsdf

    def _select_optimizer(self):
        model_optim = tf.keras.optimizers.Adam(lr=self.args.learning_rate)
        return model_optim

    tf.config.run_functions_eagerly(False)
    @tf.function
    def loss_fn(self, y_true,y_pred):
        weight_y_pred = tf.math.multiply_no_nan(y_pred, self.label_weight)
        return -tf.math.multiply_no_nan(y_true, weight_y_pred)

    def train(self):

        # model compile
        ckpt_path = os.path.join(self.checkpoint_path, 'checkpoint')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=1, save_best_only=True,
                                                        save_weights_only=True, mode='min')
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=self.args.patience, restore_best_weights=True)
        callbacks_list = [checkpoint, early_stopping]
        model_optim = self._select_optimizer()
        self.model = Informer(self.args, mode='pretrain')

        self.model.save_weights(ckpt_path)

        # fit model
        pred = []
        self.attn_f = []
        for window in tqdm(self.windows):
            pretrain, val, online = window
            fn = lambda dataset: dataset.apply(tf.data.experimental.ignore_errors())
            pretrain, val, online = fn(pretrain), fn(val), fn(online)
            self.model = Informer(self.args, mode='pretrain')
            self.model.load_weights(ckpt_path)
            self.model.compile(optimizer=model_optim, loss=self.loss_fn, run_eagerly=self.args.run_eagerly)
            self.model.fit(pretrain, validation_data=val,
                               callbacks=callbacks_list,
                               epochs=self.args.train_epochs,
                               verbose=1, shuffle=True)

            self.model = Informer(self.args, mode='online')
            self.model.load_weights(ckpt_path)
            self.model.compile(optimizer=model_optim, loss=self.loss_fn, run_eagerly=self.args.run_eagerly)

            self.model.fit(online)
            pred.append(self.model.pred)
            self.attn_f.append(self.model.features)
        self.pred = tf.concat(pred, axis=0)








