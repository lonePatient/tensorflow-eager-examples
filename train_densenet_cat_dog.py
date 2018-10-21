#encoding:utf-8
import json
import os
import numpy as np
import time
import tensorflow as tf
from pytfeager.model.densenet import DenseNet
from pytfeager.callbacks.model_callbacks import WriteSummary
from pytfeager.callbacks.model_callbacks import EarlyStopping
from pytfeager.callbacks.model_callbacks import ModelCheckpoint
from pytfeager.callbacks.model_callbacks import ProgressBar
from pytfeager.callbacks.model_callbacks import CyclicLR
from pytfeager.utils.data_utils import get_dataset
from pytfeager.config import densenet_config as config


tf.enable_eager_execution()
tf.set_random_seed(2018)
np.random.seed(2018)

def main(_):

    feature_path = os.path.join(config.PATH, config.FEATURES_PATH) # tfrecord数据路径
    log_path = os.path.join(config.PATH, config.LOG_PATH)  # 日志路径
    checkpoint_dir = os.path.join(config.PATH, config.CHECKPOINT_PATH) # checkpoint路径

    train_tfrecord = os.path.join(feature_path, 'train.tfrecords')
    test_tfrecord = os.path.join(feature_path, 'test.tfrecords')
    buffer_size = sum(1 for _ in tf.python_io.tf_record_iterator(train_tfrecord)) # 计算训练数据总数

    n_batch = buffer_size // FLAGS.batch_size
    resiud  = buffer_size % FLAGS.batch_size
    base_lr = FLAGS.max_learning_rate / 3
    stepsize = 2 * n_batch


    # 获取数据集
    train_dataset,val_dataset = get_dataset(train_tfrecord=train_tfrecord,
                                            val_tfrecord = test_tfrecord,
                                            buffer_size=1000,
                                            batch_size=FLAGS.batch_size)
    #采用cyclical机制
    learning_rate_tf = tf.Variable(base_lr) # 定义学习率张量
    clr = CyclicLR(base_lr = base_lr,
                   init_lr=learning_rate_tf,
                   max_lr=FLAGS.max_learning_rate,
                   mode='triangular',
                   step_size=stepsize)
    optimizer = tf.train.AdamOptimizer(learning_rate=clr.learning_rate,beta1=0.98,beta2=0.99)
    # 初始化模型
    model = DenseNet(k=FLAGS.growth_rate,
                     weight_decay=FLAGS.l2_regularization,
                     num_outputs=FLAGS.num_outputs,
                     units_per_block=FLAGS.units_per_block,
                     momentum=FLAGS.momentum,
                     epsilon=FLAGS.epsilon,
                     initial_pool=FLAGS.initial_pool)

    save_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'global_step': tf.train.get_or_create_global_step()
    }
    # 定义callbacks
    write_summary = WriteSummary(log_path=log_path)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=checkpoint_dir, mode='min', monitor='val_Loss',save_best_only=False, epoch_per=3,
                                       **save_kwargs)
    early_stop = EarlyStopping(mode='min',patience=FLAGS.early_stopping_rounds)
    progressbar = ProgressBar(data_size=buffer_size, n_batch=n_batch, batch_size=FLAGS.batch_size, resiud=resiud,
                              eval_name='acc', loss_name='loss')

    # 开始训练模型
    model.fit(trainDataset=train_dataset,
              valDataset=val_dataset,
              epochs=FLAGS.epochs,
              optimizer=optimizer,
              ModelCheckPoint=model_checkpoint,
              progressbar=progressbar,
              write_summary=write_summary,
              early_stopping=early_stop,
              lr_schedule=clr)

if __name__ =='__main__':

    tf.app.flags.DEFINE_float('max_learning_rate', 0.06,
                              'Maximum learning rate value.')
    tf.app.flags.DEFINE_integer('batch_size', 64,
                                'Number of training pairs per iteration.')
    tf.app.flags.DEFINE_integer('growth_rate', 32,
                                'Densenet growth_rate factor.')
    tf.app.flags.DEFINE_integer('epochs', 50,
                                'number of training')
    tf.app.flags.DEFINE_float('l2_regularization', 0.03,
                              'Weight decay regularization penalty.')
    tf.app.flags.DEFINE_integer('num_outputs', 1,
                                'Number of output units for DenseNet.')
    tf.app.flags.DEFINE_list('units_per_block', [6, 12, 24, 16],
                             'DenseNet units and blocks architecture.')
    tf.app.flags.DEFINE_float('momentum', 0.997,
                              'Momentum for batch normalization.')
    tf.app.flags.DEFINE_float('epsilon', 0.001,
                              'Epsilon for batch normalization.')
    tf.app.flags.DEFINE_bool('initial_pool', True,
                             'Should the DenseNet include the first pooling layer.')
    tf.app.flags.DEFINE_integer('early_stopping_rounds',10,'number of early stopping')
    FLAGS = tf.app.flags.FLAGS
    tf.app.run()
