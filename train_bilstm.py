#encoding:utf-8
import os
import numpy as np
import tensorflow as tf
import gc
from tensorflow.python.keras.preprocessing import sequence
from pytfeager.io.io import load_imdb
from pytfeager.model.bilstm import BiLSTM
from pytfeager.callbacks.model_callbacks import WriteSummary
from pytfeager.callbacks.model_callbacks import EarlyStopping
from pytfeager.callbacks.model_callbacks import ModelCheckpoint
from pytfeager.callbacks.model_callbacks import ProgressBar
from pytfeager.config import bilstm_config as config

tf.enable_eager_execution()
tf.set_random_seed(2018)
np.random.seed(2018)


def main(_):

    log_path = os.path.join(config.PATH, config.LOG_PATH)
    checkpoint_dir = os.path.join(config.PATH, config.CHECKPOINT_PATH)

    # 加载数据集
    (x_train, y_train), (x_test, y_test) = load_imdb(path=os.path.join(config.PATH,config.DATA_PATH),
                                                        maxlen=FLAGS.maxlen,
                                                        num_words=FLAGS.num_words)

    train_sequence = sequence.pad_sequences(x_train, maxlen=FLAGS.maxlen, dtype='int32',
                                padding='post', truncating='post', value=0)
    test_sequence = sequence.pad_sequences(x_test, maxlen=FLAGS.maxlen, dtype='int32',
                                padding='post', truncating='post', value=0)


    gc.collect()

    buffer_size = len(train_sequence)
    n_batch = buffer_size // FLAGS.batch_size
    resiud  = buffer_size % FLAGS.batch_size

    train_dataset = tf.data.Dataset.from_tensor_slices((train_sequence, y_train)).shuffle(10000)
    train_dataset = train_dataset.batch(FLAGS.batch_size, drop_remainder=False)

    val = [tf.convert_to_tensor(test_sequence), tf.convert_to_tensor(y_test)]

    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    # 初始化模型
    model = BiLSTM(units=FLAGS.lstm_units,
                          output_dim=FLAGS.output_dim,
                          input_length=FLAGS.maxlen,
                          input_dim=FLAGS.num_words,
                          trainable=True)
    save_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'global_step': tf.train.get_or_create_global_step()
    }
    # 定义callbacks
    write_summary = WriteSummary(log_path=log_path)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=checkpoint_dir, mode='min', monitor='val_loss',
                                       save_best_only=True, epoch_per=None,
                                       **save_kwargs)
    early_stop = EarlyStopping(mode='min',patience=FLAGS.early_stopping_rounds)
    progressbar = ProgressBar(data_size=buffer_size, n_batch=n_batch, batch_size=FLAGS.batch_size, resiud=resiud,
                              eval_name='acc', loss_name='loss')
    # 开始训练模型
    model.fit(trainDataset=train_dataset,
              valDataset=val,
              epochs=FLAGS.epochs,
              optimizer=optimizer,
              ModelCheckPoint=model_checkpoint,
              progressbar=progressbar,
              write_summary=write_summary,
              early_stopping=early_stop)

if __name__ =='__main__':
    tf.app.flags.DEFINE_float('learning_rate', 0.01,
                              'learning rate value.')
    tf.app.flags.DEFINE_integer('batch_size', 64,
                                'Number of training pairs per iteration.')
    tf.app.flags.DEFINE_integer('epochs',20,
                                'number of training')
    tf.app.flags.DEFINE_integer('early_stopping_rounds',10,
                                'number of early stopping')
    tf.app.flags.DEFINE_integer('lstm_units',64,
                                'cells number of lstm')
    tf.app.flags.DEFINE_integer('maxlen',600,
                                'max length of sequence')
    tf.app.flags.DEFINE_integer('output_dim',32,
                                'output dim of embedding')
    tf.app.flags.DEFINE_integer('num_words',10000,
                                'word number')
    FLAGS = tf.app.flags.FLAGS
    tf.app.run()

