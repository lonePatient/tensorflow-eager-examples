#encoding:utf-8
import os
import numpy as np
import time
import tensorflow as tf
from pytfeager.io.io import load_mnist
from pytfeager.model.vae import VAE
from pytfeager.utils.loss_utils import reconstruction_loss,KL_divergence
from pytfeager.callbacks.model_callbacks import WriteSummary
from pytfeager.callbacks.model_callbacks import EarlyStopping
from pytfeager.callbacks.model_callbacks import ModelCheckpoint
from pytfeager.callbacks.model_callbacks import ProgressBar
from pytfeager.config import vae_config as config

tf.enable_eager_execution()
tf.set_random_seed(2018)
np.random.seed(2018)

def main(_):

    image_size = 28 * 28
    log_path = os.path.join(config.PATH, config.LOG_PATH)
    checkpoint_dir = os.path.join(config.PATH, config.CHECKPOINT_PATH)

    # 加载数据集
    (x_train, y_train), (x_test, _) = load_mnist(path=os.path.join(config.PATH,config.DATA_PATH))

    x_train = x_train.astype('float32') / 255
    x_test  = x_test.astype('float32') / 255
    x_train = tf.reshape(x_train,shape=(-1,image_size))
    x_test  = tf.reshape(x_test,shape=(-1,image_size))


    buffer_size = x_train.numpy().shape[0]
    n_batch = buffer_size // FLAGS.batch_size
    resiud  = buffer_size % FLAGS.batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, )).shuffle(1000)
    train_dataset = train_dataset.batch(FLAGS.batch_size, drop_remainder=False)

    #优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    # 初始化模型
    model = VAE(h_dim=FLAGS.h_dim,z_dim=FLAGS.z_dim,image_zise=image_size)
    save_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'global_step': tf.train.get_or_create_global_step()
    }
    # 定义callbacks
    write_summary = WriteSummary(log_path=log_path)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=checkpoint_dir, mode='min',monitor='val_loss', save_best_only=False, epoch_per=3,
                                       **save_kwargs)
    early_stop = EarlyStopping(mode='min',patience=FLAGS.early_stopping_rounds)
    progressbar = ProgressBar(data_size=buffer_size, n_batch=n_batch, batch_size=FLAGS.batch_size, resiud=resiud,
                              eval_name='acc', loss_name='loss')

    # 开始训练模型
    model.fit(trainDataset=train_dataset,
              valDataset=[x_test,_],
              epochs=FLAGS.epochs,
              optimizer=optimizer,
              ModelCheckPoint=model_checkpoint,
              progressbar=progressbar,
              write_summary=write_summary,
              early_stopping=early_stop)

if __name__ =='__main__':

    tf.app.flags.DEFINE_float('learning_rate', 0.001,'learning rate value.')
    tf.app.flags.DEFINE_integer('batch_size', 500,'Number of training pairs per iteration.')
    tf.app.flags.DEFINE_integer('epochs',15,'number of training')
    tf.app.flags.DEFINE_integer('early_stopping_rounds',10,'number of early stopping')
    tf.app.flags.DEFINE_integer('h_dim', 512, 'h_dim')
    tf.app.flags.DEFINE_integer('z_dim', 5, 'z_dim')
    FLAGS = tf.app.flags.FLAGS
    tf.app.run()

