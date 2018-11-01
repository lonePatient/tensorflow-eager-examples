#encoding:utf-8
import os
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import spline
plt.switch_backend('agg') # 防止ssh上绘图问题

class LrFinder():

    def __init__(self, model,data_size,batch_szie,fig_path):

        # 进度条变量
        self.data_size = data_size
        self.batch_size = batch_szie
        self.loss_name = 'loss'
        self.eval_name = 'acc'
        self.width = 30
        self.resiud = self.data_size % self.batch_size
        self.n_batch = self.data_size // self.batch_size

        # lr_find变量
        self.model = model
        self.best_loss = 1e9
        self.fig_path = fig_path
        self.losses = []
        self.lrs = []

        self.model_status = False
        self.learning_rate = tf.Variable(0.001,trainable=False)


    # 每次batch更新进行操作
    def on_batch_end(self, loss,):
        # 更新学习率和loss列表数据
        self.lrs.append(self.learning_rate.numpy())
        self.losses.append(loss.numpy())

        # 对loss进行判断，主要当loss增加时，停止训练
        if  loss.numpy() > self.best_loss * 4:
            self.model_status = True

        #　更新best_loss
        if loss.numpy() < self.best_loss:
            self.best_loss = loss.numpy()

        # 学习率更新方式
        lr = self.lr_mult * self.learning_rate.numpy()
        self.learning_rate.assign(lr)

    # 自定义的进度条信息
    def progressbar(self,batch,loss,acc):
        recv_per = int(100 * (batch + 1) / self.n_batch)
        if batch == self.n_batch:
            num = batch * self.batch_size + self.resiud
        else:
            num = batch * self.batch_size
        if recv_per >= 100:
            recv_per = 100
        # 字符串拼接的嵌套使用
        show_bar = ('[%%-%ds]' % self.width) % (int(self.width * recv_per / 100) * ">")
        show_str = '\r%d/%d %s - %s: %.4f- %s: %.4f'
        print(show_str % (
            num, self.data_size,
            show_bar,
            self.loss_name, loss,
            self.eval_name, acc ),
                end='')

    # 对loss进行可视化
    def plot_loss(self):
        plt.style.use("ggplot")
        plt.figure()
        plt.ylabel("loss")
        plt.xlabel("learning rate")
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
        plt.savefig(os.path.join(self.fig_path,'loss.jpg'))
        plt.close()

    # 这里我们使用scipy模块简单点进行平滑化，方便查看
    def plot_loss_smooth(self):
        #这里我们采用移动平均进行平滑化
        xnew = np.linspace(min(self.lrs),max(self.lrs),100)
        smooth_loss = spline(self.lrs,self.losses,xnew)
        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(xnew, smooth_loss)
        plt.xscale('log')
        plt.savefig(os.path.join(self.fig_path,'smooth_loss.jpg'))
        plt.close()

    # 定义一个拟合函数
    def find(self, trainDatsset, start_lr, end_lr, optimizer,epochs=1,verbose = 1,save = True):
        self.learning_rate.assign(start_lr)

        num_batches = epochs * self.data_size / self.batch_size

        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))

        for i in range(epochs):
            print("\nEpoch {i}/{epochs}".format(i=i + 1, epochs=epochs))
            for (batch_id, (X, y)) in enumerate(trainDatsset):
                y_pred,train_loss,grads = self.model.compute_grads(X, y)
                train_score = self.model.compute_score(y_pred=y_pred,y_true=y)
                optimizer.apply_gradients(zip(grads, self.model.variables))

                self.on_batch_end(loss=train_loss)
                if verbose > 0 :
                    self.progressbar(batch=batch_id, loss=train_loss,acc=train_score)
                if self.model_status:
                    break

            if self.model_status:
                break
        if save:
            self.plot_loss()
            self.plot_loss_smooth()



