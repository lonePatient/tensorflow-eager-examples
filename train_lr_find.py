#encoding:utf-8
import os
import tensorflow as tf
import numpy as np
from pytfeager.io.io import load_mnist
from pytfeager.model.cnn import CNN
from pytfeager.config import cnn_config as config
from pytfeager.callbacks.lr_finder import LrFinder

tf.enable_eager_execution()
tf.set_random_seed(2018)
np.random.seed(2018)

def main(_):

    fig_path = '.'
    # 加载数据
    (x_train, y_train), (_, _) = load_mnist(path=os.path.join(config.PATH,config.DATA_PATH))

    x_train = x_train.astype('float32') / 255.
    x_train = tf.reshape(x_train,shape=(-1,28,28,1))

    y_train_ohe = tf.one_hot(y_train,depth=FLAGS.num_classes)

    buffer_size = x_train.numpy().shape[0]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_ohe))


    train_dataset = train_dataset.shuffle(1000)
    train_dataset = train_dataset.batch(FLAGS.batch_size, drop_remainder=False)

    # 初始化模型
    model = CNN(num_classes = FLAGS.num_classes)

    lr_finder = LrFinder(model = model,data_size=buffer_size,batch_szie= FLAGS.batch_size,
                         fig_path = fig_path)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_finder.learning_rate)
    lr_finder.find(trainDatsset=train_dataset,start_lr=0.00001,end_lr = 10,optimizer=optimizer,epochs=1)

if __name__ =='__main__':

    tf.app.flags.DEFINE_float('learning_rate', 0.001,
                              'learning rate value.')
    tf.app.flags.DEFINE_integer('batch_size', 64,
                                'Number of training pairs per iteration.')
    tf.app.flags.DEFINE_integer('epochs',50,
                                'number of training')
    tf.app.flags.DEFINE_integer('early_stopping_rounds',10,
                                'number of early stopping')
    tf.app.flags.DEFINE_integer('num_classes',10,
                                'class number of mnist data')
    tf.app.flags.DEFINE_bool('data_aug',False,
                             'use data aug')
    FLAGS = tf.app.flags.FLAGS
    tf.app.run()


