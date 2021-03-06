#encoding:utf-8
import time
import numpy as np
import tensorflow as tf
from pytfeager.utils.loss_utils import sigmoid_cross_entropy_loss
from pytfeager.utils.eval_utils import binary_accuracy

class RNN(tf.keras.Model):
    def __init__(self,units,output_dim,input_dim,input_length,trainable):
        super(RNN,self).__init__()
        self.embedding = tf.keras.layers.Embedding(output_dim=output_dim,
                                                   input_dim=input_dim,
                                                   input_length=input_length,
                                                   trainable=trainable)
        self.rnn = tf.keras.layers.LSTM(units,unroll=True) # slower if not unrolled
        self.fc = tf.keras.layers.Dense(1)

    def call(self,inputs):

        x = self.embedding(inputs)
        x = self.rnn(x)
        x = self.fc(x)
        y = tf.nn.sigmoid(x)
        return y

    def compute_loss(self,X,y):
        y_pred = self(X)[:,0]
        loss = sigmoid_cross_entropy_loss(y_pred=y_pred,y_true = y)
        return y_pred,loss

    def compute_grads(self,X,y):
        with tf.GradientTape() as tape:
            y_pred,loss = self.compute_loss(X, y)
        return y_pred,loss,tape.gradient(loss, self.variables)

    def compute_score(self,y_pred,y_true):
        score = binary_accuracy(y_pred=y_pred,y_true=y_true)
        return score

    def restore_model(self,ModelCheckPoint):
        # 是否加载模型继续训练
        model_path = ModelCheckPoint.checkpoint_dir
        ModelCheckPoint.checkpoint.restore(tf.train.latest_checkpoint(model_path))

    def fit(self,trainDataset,valDataset,epochs,optimizer,ModelCheckPoint,progressbar,
            write_summary,early_stopping=None,verbose=1,train_from_scratch=False):
        # 是否加载模型继续训练
        if train_from_scratch == True:
            self.restore_model(ModelCheckPoint)

        for i in range(epochs):
            print("\nEpoch {i}/{epochs}".format(i=i + 1, epochs=epochs))
            loss_list = []
            score_list = []
            for (ind, (X, y)) in enumerate(trainDataset):
                start = time.time()
                y_pred,train_loss,grads = self.compute_grads(X, y)
                train_score = self.compute_score(y_pred=y_pred,y_true=y)
                optimizer.apply_gradients(zip(grads, self.variables))

                loss_list.append(train_loss)
                score_list.append(train_score)

                # 计算 验证集
                y_pred,val_loss = self.compute_loss(X=valDataset[0], y=valDataset[1])
                val_score = self.compute_score(y_pred = y_pred, y_true=valDataset[1])

                progressbar.on_batch_end(batch=ind, loss=train_loss,
                                         acc=train_score, val_loss=val_loss,
                                         val_acc=val_score, use_time=time.time() - start)

            # 日志记录
            write_summary.on_epoch_end(tensor=np.mean(loss_list), name='train_loss', step=i)
            write_summary.on_epoch_end(tensor=val_loss, name='val_loss', step=i)
            write_summary.on_epoch_end(tensor=np.mean(score_list), name='train_acc', step=i)
            write_summary.on_epoch_end(tensor=val_score, name='val_acc', step=i)

            if early_stopping:
                early_stopping.on_epoch_end(epoch=i, current=val_loss)
                if early_stopping.stop_training:
                    break
            ModelCheckPoint.on_epoch_end(epoch=i, current=val_loss)