#encoding:utf-8
import os
import tensorflow as tf
import numpy as np
from pytfeager.io.io import load_mnist
from pytfeager.model.cnn import CNN
from pytfeager.callbacks.model_callbacks import WriteSummary
from pytfeager.callbacks.model_callbacks import EarlyStopping
from pytfeager.callbacks.model_callbacks import ModelCheckpoint
from pytfeager.callbacks.model_callbacks import ProgressBar
from pytfeager.preprocessing.data_preocessing import random_brightness_image
from pytfeager.preprocessing.data_preocessing import central_scale_images
from pytfeager.config import cnn_config as config

tf.enable_eager_execution()
tf.set_random_seed(2018)
np.random.seed(2018)

def main(_):

    log_path = os.path.join(config.PATH, config.LOG_PATH)
    checkpoint_dir = os.path.join(config.PATH, config.CHECKPOINT_PATH)

    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_mnist(path=os.path.join(config.PATH,config.DATA_PATH))

    # cpu下好像不能使用tf.float64
    x_train = x_train.astype('float32') / 255.
    x_test  = x_test.astype('float32') / 255.
    x_train = tf.reshape(x_train,shape=(-1,28,28,1))
    x_test  = tf.reshape(x_test,shape = (-1,28,28,1))

    y_train_ohe = tf.one_hot(y_train,depth=FLAGS.num_classes)
    y_test_ohe  = tf.one_hot(y_test, depth=FLAGS.num_classes)

    buffer_size = x_train.numpy().shape[0]
    n_batch = buffer_size // FLAGS.batch_size
    resiud  = buffer_size % FLAGS.batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_ohe))

    if FLAGS.data_aug:
        '''
        好像dataset里面不太好实用numpy形式（调用cv2）进行数据曾强处理
        '''
        c_i_dataset = train_dataset.map(lambda img,label:[central_scale_images(img,scale=0.65),label])
        r_b_dataset = train_dataset.map(lambda img, label: [random_brightness_image(img), label])
        train_dataset = train_dataset.concatenate(r_b_dataset)
        train_dataset = train_dataset.concatenate(c_i_dataset)
        buffer_size = sum([1 for _ in train_dataset])

    train_dataset = train_dataset.shuffle(1000)
    train_dataset = train_dataset.batch(FLAGS.batch_size, drop_remainder=False)

    #优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    # 初始化模型
    model = CNN(num_classes = FLAGS.num_classes)

    save_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'global_step': tf.train.get_or_create_global_step()
    }

    # 定义callbacks
    write_summary = WriteSummary(log_path=log_path)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=checkpoint_dir, mode='min',
                                       monitor='val_loss',save_best_only=True,
                                       epoch_per=3,**save_kwargs)

    early_stop = EarlyStopping(mode='min',patience=FLAGS.early_stopping_rounds)
    progressbar = ProgressBar(data_size=buffer_size, n_batch=n_batch,
                              batch_size=FLAGS.batch_size, resiud=resiud,
                              eval_name='acc', loss_name='loss')

    # 开始训练模型
    model.fit(trainDataset=train_dataset,
              valDataset=[x_test,y_test_ohe],
              epochs=FLAGS.epochs,
              optimizer=optimizer,
              ModelCheckPoint=model_checkpoint,
              progressbar=progressbar,
              write_summary=write_summary,
              early_stopping=early_stop)


if __name__ =='__main__':

    tf.app.flags.DEFINE_float('learning_rate', 0.001,
                              'learning rate value.')
    tf.app.flags.DEFINE_integer('batch_size', 512,
                                'Number of training pairs per iteration.')
    tf.app.flags.DEFINE_integer('epochs',100,
                                'number of training')
    tf.app.flags.DEFINE_integer('early_stopping_rounds',10,
                                'number of early stopping')
    tf.app.flags.DEFINE_integer('num_classes',10,
                                'class number of mnist data')
    tf.app.flags.DEFINE_bool('data_aug',False,
                             'use data aug')
    FLAGS = tf.app.flags.FLAGS
    tf.app.run()


