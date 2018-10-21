#encoding:utf-8
import time
import numpy as np
import tensorflow as tf
from pytfeager.utils.loss_utils import softmax_cross_entropy_loss
from pytfeager.utils.eval_utils import categorical_accuracy

class ConvBnRelu(tf.keras.Model):
    def __init__(self,filters,strides = 1,kernel = 3,padding = 'same'):
        super(ConvBnRelu,self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters = filters, # 过滤器
                                           kernel_size=(kernel,kernel), # 核大小
                                           strides=(strides,strides),   # 步长
                                           padding=padding,             # padding方式
                                           use_bias=False,              #偏差
                                           kernel_initializer='he_normal')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self,inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = tf.nn.relu(x)
        return x

class InceptionBlock(tf.keras.Model):
    def __init__(self,filters,strides=1):
        super(InceptionBlock,self).__init__()

        self.conv1 = ConvBnRelu(filters,strides,kernel=1)
        self.conv2 = ConvBnRelu(filters,strides,kernel = 3)
        self.conv3_1 = ConvBnRelu(filters,strides,kernel = 3)
        self.conv3_2 = ConvBnRelu(filters,strides = 1,kernel = 3)
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')
        self.maxpool_conv = ConvBnRelu(filters,strides,kernel=1)

    def call(self,inputs):

        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)

        x3_1 = self.conv3_1(inputs)
        x3_2 = self.conv3_2(x3_1)

        x4 = self.maxpool(inputs)
        x4 = self.maxpool_conv(x4)

        x = tf.keras.layers.concatenate([x1,x2,x3_2,x4],axis = -1)
        return x


class InceptionNet(tf.keras.Model):
    def __init__(self,num_layers,num_classes,initial_filters=16,**kwargs):
        super(InceptionNet,self).__init__()

        self.in_channels = initial_filters
        self.out_channels = initial_filters
        self.num_layers = num_layers
        self.initial_filters = initial_filters

        self.conv1 = ConvBnRelu(initial_filters)
        self.blocks = []
        # 建立所有的block
        for block_id in range(num_layers):
            for layer_id in range(2):
                key = 'block_%d_%d'%(block_id+1,block_id+1)
                if layer_id == 0:
                    block = InceptionBlock(self.out_channels,strides=2)
                else:
                    block = InceptionBlock(self.out_channels)

                self.in_channels = self.out_channels
                # setattr(object, name, value)
                # 设置对象的属性的值
                # 该对象必须存在
                setattr(self,key,block)

                self.blocks.append(block)

            self.out_channels *= 2

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout  = tf.keras.layers.Dropout(0.5)
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self,inputs):
        out = self.conv1(inputs)
        for block in self.blocks:
            out = block(out)
        out = self.avg_pool(out)
        out = self.dropout(out)
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


