#encoding:utf-8
import time
import numpy as np
import tensorflow as tf
from pytfeager.utils.loss_utils import softmax_cross_entropy_loss
from pytfeager.utils.eval_utils import categorical_accuracy

def conv3x3(filters,stride = 1,kernel = 3):
    return tf.keras.layers.Conv2D(filters = filters,
                                  kernel_size=(kernel,kernel),
                                  strides=(stride,stride),
                                  padding='same',
                                  kernel_initializer=tf.variance_scaling_initializer())

class ResnetBlock(tf.keras.Model):
    def __init__(self,filters,strides=1,residual_path=False):
        super(ResnetBlock,self).__init__()
        self.residual_path = residual_path
        self.conv1 = conv3x3(filters,strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = conv3x3(filters)
        self.bn2 = tf.keras.layers.BatchNormalization()

        if residual_path:
            self.down_conv = conv3x3(filters,strides,kernel = 1)
            self.down_bn = tf.keras.layers.BatchNormalization()

    def call(self,inputs,training=None):
        residual = inputs

        x = self.bn1(inputs,training)
        x = tf.nn.relu(x)
        x = self.conv1(x)

        x = self.bn2(x,training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        if self.residual_path:
            residual = self.down_conv(inputs)
            residual = tf.nn.relu(residual)
            residual = self.down_bn(residual,training)

        x = x + residual

        return x

class ResNet(tf.keras.Model):
    def __init__(self,block_list,num_classes,initial_filters = 16,**kwargs):
        super(ResNet,self).__init__()

        self.num_blocks  =len(block_list)
        self.block_list = block_list

        self.in_channels = initial_filters
        self.out_channels = initial_filters
        self.conv_initial = conv3x3(self.out_channels)

        self.blocks = []

        for block_id in range(self.num_blocks):
            for layer_id in range(block_list[block_id]):
                key = 'block_%d_%d' % (block_id +1,block_id+1)
                if block_id != 0 and layer_id ==0:
                    block = ResnetBlock(self.out_channels,strides = 2,residual_path=True)
                else:
                    if self.in_channels != self.out_channels:
                        residual_path = True
                    else:
                        residual_path = False
                    block = ResnetBlock(self.out_channels,residual_path=residual_path)
                self.in_channels = self.out_channels
                setattr(self,key,block)
                self.blocks.append(block)

            self.out_channels *= 2
        self.final_bn = tf.keras.layers.BatchNormalization()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def forward(self,inputs,training=None):
        out = self.conv_initial(inputs)

        for block in self.blocks:
            out = block(out)
        out = self.final_bn(out,training)
        out = tf.nn.relu(out)

        out = self.avg_pool(out)
        out = self.fc(out)

        out = tf.nn.softmax(out)

        return out

    def compute_loss(self,X,y):
        y_pred = self(X)
        loss = softmax_cross_entropy_loss(y_pred=y_pred,y_true = y)
        return y_pred,loss

    def compute_grads(self,X,y):
        with tf.GradientTape() as tape:
            y_pred,loss = self.compute_loss(X, y)
        return y_pred,loss,tape.gradient(loss, self.variables)

    def compute_score(self,y_pred,y_true):
        score = categorical_accuracy(y_pred=y_pred,y_true=y_true)
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
                if verbose >0:
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