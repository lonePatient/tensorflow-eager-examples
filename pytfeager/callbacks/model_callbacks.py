#encoding:utf-8
import os
import numpy as np
import json
import tensorflow as tf

'''
keras的训练模式

a. model.fit_generator 训练入口函数(参考上面的函数原型定义)， 我们项目中用tk_data_generator函数作为训练数据提供者（生成器）
1) callbacks.on_train_begin()
2) while epoch < epochs:
3)         callbacks.on_epoch_begin(epoch)
4)         while steps_done < steps_per_epoch:
5）             generator_output = next(output_generator)       #生成器next函数取输入数据进行训练，每次取一个batch大小的量
6）             callbacks.on_batch_begin(batch_index, batch_logs)
7）             outs = self.train_on_batch(x, y,sample_weight=sample_weight,class_weight=class_weight)
8）             callbacks.on_batch_end(batch_index, batch_logs)
            end of while steps_done < steps_per_epoch
            self.evaluate_generator(...)          #当一个epoch的最后一次batch执行完毕，执行一次训练效果的评估
9）      callbacks.on_epoch_end(epoch, epoch_logs)          #在这个执行过程中实现模型数据的保存操作
      end of while epoch < epochs
10) callbacks.on_train_end()

所以callbacks主要是从keras源码中获取的。
'''


class ReduceLROnPlateau(object):
    '''
    monitor：被监测的量
    factor：每次减少学习率的因子，学习率将以lr = lr * factor的形式被减少
    patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    mode：‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
    epsilon：阈值，用来确定是否进入检测值的“平原区”
    cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
    min_lr：学习率的下限
    '''
    def __init__(self, init_lr,factor=0.1, patience=10,
                 verbose=1, mode='min', min_delta=1e-4, cooldown=0, min_lr=0.00001,
                 **kwargs):
        self.learning_rate = init_lr
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode == 'min' :
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0

    def on_epoch_end(self, epoch, current):
        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = self.learning_rate
                if tf.greater(old_lr,self.min_lr):
                    new_lr = old_lr * self.factor
                    new_lr = tf.maximum(new_lr, self.min_lr)
                    self.learning_rate.assign(new_lr)
                    if self.verbose > 0:
                        print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                'rate to %s.' % (epoch + 1, new_lr.numpy()))
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

class CyclicLR(object):
    def __init__(self, base_lr, init_lr,max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        self.base_lr = base_lr
        self.learning_rate =init_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

    def on_batch_end(self, epoch):

        self.trn_iterations += 1
        self.clr_iterations += 1
        self.learning_rate.assign(self.clr())


class EarlyStopping(object):
    def __init__(self,
                 min_delta=0,
                 patience=10,
                 verbose=0,
                 mode='min',
                 baseline=None):
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
        self._reset()

    def _reset(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, current):
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True

class ModelCheckpoint(object):
    '''
    由于tensorflow不知道怎么保存最好文件，
    所以如果选择保存最好模型，则我这里先清空文件夹

    另一种是每个几个epoch保存一次
    '''
    def __init__(self, checkpoint_dir,monitor,
                 save_best_only=False,
                 mode='min', epoch_per=1,
                 **kwargs):
        self.monitor = monitor
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.save_best_only = save_best_only
        self.epoch_per = epoch_per
        self.checkpoint = tf.train.Checkpoint(**kwargs)

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf

        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def on_epoch_end(self, epoch,current,save_meta=None):

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                print('\nEpoch %05d: %s improved from %0.5f to %0.5f'% (epoch + 1, self.monitor, self.best,
                         current))
                self.best = current
                command = 'rm -rf %s/*.index %s/*.data* %s/checkpoint'%(self.checkpoint_dir,
                                                                      self.checkpoint_dir,
                                                                      self.checkpoint_dir)
                os.system(command=command)
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                if save_meta is not None:
                    with open(self.checkpoint_dir+'meta.json','w') as fp:
                        json.dump(save_meta,fp,sort_keys = True,indent=4)
        else:
            if epoch % self.epoch_per == 0:
                print("\nEpoch %05d: save model to disk."%(epoch+1))
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                if save_meta is not None:
                    with open(self.checkpoint_dir + 'meta.json', 'w') as fp:
                        json.dump(save_meta, fp, sort_keys=True, indent=4)



class WriteSummary(object):
    '''
    日志记录器，方便后续tensorboard查看
    '''

    def __init__(self,log_path):
        self.log_path = log_path

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.writer = tf.contrib.summary.create_file_writer(self.log_path)

    def on_epoch_end(self,tensor,name,step):
        with tf.contrib.summary.always_record_summaries(): # record_summaries_every_n_global_steps(1)
            self.writer.set_as_default()
            tf.contrib.summary.scalar(name=name, tensor=tensor,step=step)

class ProgressBar():
    '''
     按照keras的进度条显示模式
    '''

    def __init__(self,
                 data_size,
                 n_batch,
                 batch_size,
                 resiud,
                 eval_name='acc',
                 loss_name='loss',
                 width=30
                 ):
        self.width = width
        self.loss_name = loss_name
        self.eval_name = eval_name
        self.data_size  = data_size
        self.n_batch = n_batch
        self.batch_size = batch_size
        self.resiud = resiud

    def on_batch_end(self,batch,
                 loss,
                 val_loss,
                 acc,
                 val_acc,
                 use_time):
        recv_per = int(100 * (batch + 1) / self.n_batch)
        if batch == self.n_batch:
            num = batch * self.batch_size + self.resiud
        else:
            num = batch * self.batch_size
        if recv_per >= 100:
            recv_per = 100
        show_bar = ('[%%-%ds]' % self.width) % (int(self.width * recv_per / 100) * ">")  # 字符串拼接的嵌套使用
        show_str = '\r%d/%d %s -%.1fs/step- %s: %.4f- %s: %.4f - val_%s: %.4f - val_%s: %.4f'
        print(show_str % (
            num, self.data_size,
            show_bar, use_time,
            self.loss_name, loss,
            self.eval_name, acc,
            self.loss_name, val_loss,
            self.eval_name, val_acc),
                end='')
