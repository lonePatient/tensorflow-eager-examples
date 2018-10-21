#encoding:utf-8
import os
import numpy as np
import tensorflow as tf
from pytfeager.io.io import load_boston
from pytfeager.preprocessing.data_preocessing import z_score
from pytfeager.model.regressor import Regressor
from pytfeager.callbacks.model_callbacks import WriteSummary
from pytfeager.callbacks.model_callbacks import EarlyStopping
from pytfeager.callbacks.model_callbacks import ModelCheckpoint
from pytfeager.callbacks.model_callbacks import ProgressBar
from pytfeager.config import linear_regression_config as config

tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)

def main(_):

    log_path = os.path.join(config.PATH,config.LOG_PATH)
    checkpoint_dir = os.path.join(config.PATH, config.CHECKPOINT_PATH)


    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_boston(path=os.path.join(config.PATH,config.DATA_PATH),
                                                        test_split=0.2,
                                                        seed=0)
    # 数据预处理，标准化
    x_train, x_test = z_score(x_train,x_test)

    # 如果全量数据不大，可以对全量数据进行shuffle
    # 如果全量数据很大，直接全量数据shuffle会很慢，
    # 则直接shuffle所需大小即可，一般都是10000
    buffer_size = len(x_train)
    resiud = buffer_size % FLAGS.batch_size
    n_batch = buffer_size // FLAGS.batch_size # batch的次数

    # 转换为dataset形式
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size)
    train_dataset = train_dataset.batch(FLAGS.batch_size, drop_remainder=False)

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    # 初始化模型
    model = Regressor()
    save_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'global_step': tf.train.get_or_create_global_step()
    }
    # 定义callbacks
    write_summary = WriteSummary(log_path=log_path)
    model_checkpoint = ModelCheckpoint(checkpoint_dir = checkpoint_dir,mode='min',monitor='val_loss',
                                       save_best_only=False,epoch_per=3,**save_kwargs)
    early_stop = EarlyStopping(mode='min',patience=FLAGS.early_stopping_rounds)
    progressbar = ProgressBar(data_size=buffer_size,n_batch=n_batch,batch_size=FLAGS.batch_size,resiud=resiud,
                              eval_name='r2',loss_name='loss')

    # 开始训练模型
    model.fit(trainDataset=train_dataset,
              valDataset=[tf.convert_to_tensor(x_test),tf.convert_to_tensor(y_test)],
              epochs=FLAGS.epochs,
              optimizer=optimizer,
              ModelCheckPoint=model_checkpoint,
              progressbar=progressbar,
              write_summary=write_summary,
              early_stopping=early_stop)



if __name__ =='__main__':
    tf.app.flags.DEFINE_float('learning_rate', 0.05,
                              'learning rate value.')
    tf.app.flags.DEFINE_integer('batch_size', 128,
                                'Number of training pairs per iteration.')
    tf.app.flags.DEFINE_integer('epochs',100,
                                'number of training')
    tf.app.flags.DEFINE_integer('early_stopping_rounds',10,
                                'number of early stopping')
    FLAGS = tf.app.flags.FLAGS

    tf.app.run()
