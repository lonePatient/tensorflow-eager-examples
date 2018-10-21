#encoding:utf-8
import time
import numpy as np
import tensorflow as tf
from pytfeager.utils.loss_utils import reconstruction_loss,KL_divergence

class VAE(tf.keras.Model):

    def __init__(self,h_dim,z_dim,image_zise):
        super(VAE,self).__init__()
        self.image_size = image_zise
        self.fc1 = tf.keras.layers.Dense(h_dim)
        self.fc2 = tf.keras.layers.Dense(z_dim)
        self.fc3 = tf.keras.layers.Dense(z_dim)
        self.fc4 = tf.keras.layers.Dense(h_dim)
        self.fc5 = tf.keras.layers.Dense(self.image_zise)

    def encode(self,x):
        h = self.fc1(x)
        h = tf.nn.relu(h)
        return self.fc2(h),self.fc3(h)

    def reparameterize(self,mu,log_var):
        std = tf.exp(log_var * 0.5)
        eps = tf.random_normal(std.shape)
        return mu + eps * std

    def decode_logits(self,z):
        h = self.fc4(z)
        h = tf.nn.relu(h)
        h = self.fc5(h)
        return h

    def decode(self,z):
        z = self.decode_logits(z)
        z = tf.nn.sigmoid(z)
        return z

    def call(self,inputs):

        mu,log_var= self.encode(inputs)
        z = self.reparameterize(mu,log_var)
        x_reconstructed_logits = self.decode_logits(z)
        return x_reconstructed_logits,mu,log_var

    def compute_loss(self,X,y):
        x_reconstruction_logits, mu, log_var = self(X)
        reconst_loss = reconstruction_loss(image_size=self.image_size,
                                           x_true=X,
                                           x_pred=x_reconstruction_logits)
        kl_div = KL_divergence(log_var=log_var, mu=mu)
        train_loss = tf.reduce_mean(reconst_loss) + kl_div
        return kl_div,train_loss

    def compute_grads(self,X,y):
        with tf.GradientTape() as tape:
            kl_div, train_loss = self.compute_loss(X, y)
        return kl_div,train_loss,tape.gradient(train_loss, self.variables)

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
            total_kl_div = []
            for (ind, (X, y)) in enumerate(trainDataset):
                start = time.time()
                kl_div, train_loss,grads = self.compute_grads(X, y)
                optimizer.apply_gradients(zip(grads, self.variables))

                loss_list.append(train_loss)
                total_kl_div.append(kl_div)

                # 计算 验证集
                val_kl_div, val_loss = self.compute_loss(X=valDataset[0], y=None)

                progressbar.on_batch_end(batch=ind, loss=tf.reduce_mean(train_loss),
                                         acc=tf.reduce_sum(kl_div), val_loss=tf.reduce_mean(val_loss),
                                         val_acc=tf.reduce_sum(val_kl_div), use_time=time.time() - start)

            # 日志记录
            write_summary.on_epoch_end(tensor=np.mean(tf.reduce_mean(loss_list)), name='train_loss', step=i)
            write_summary.on_epoch_end(tensor=tf.reduce_mean(val_loss), name='val_loss', step=i)

            if early_stopping:
                early_stopping.on_epoch_end(epoch=i, current=tf.reduce_mean(val_loss))
                if early_stopping.stop_training:
                    break
            ModelCheckPoint.on_epoch_end(epoch=i, current=tf.reduce_mean(val_loss))

