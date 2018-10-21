#encoding:utf-8
import time
import numpy as np
import tensorflow as tf
from pytfeager.utils.loss_utils import sigmoid_cross_entropy_loss
from pytfeager.utils.eval_utils import binary_accuracy

class TransitionLayer(tf.keras.Model):
    def __init__(self,theta,
                 depth,
                 weight_decay,
                 momentum=0.99,
                 epsilon = 0.001):
        super(TransitionLayer,self).__init__()

        self.l2 = tf.keras.regularizers.l2
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=momentum,epsilon=epsilon)
        self.conv1 = tf.keras.layers.Conv2D(filters=int(theta * depth),
                                            use_bias = False,
                                            kernel_size=(1,1),
                                            activation=None,
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=self.l2(weight_decay),
                                            strides = (1,1),
                                            padding='same')

        self.pool1 = tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='same')

    def call(self,inputs,training):
        net = self.bn1(inputs,training=training)
        net = tf.nn.relu(net)
        net = self.conv1(net)
        net = self.pool1(net)
        return net

class DenseUnit(tf.keras.Model):
    def __init__(self,k,
                 weight_decay,
                 momentum = 0.99,
                 epsilon = 0.001):
        super(DenseUnit,self).__init__()

        self.l2 = tf.keras.regularizers.l2
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=momentum,epsilon=epsilon)
        self.conv1 = tf.keras.layers.Conv2D(filters=4 * k,
                                            use_bias= False,
                                            kernel_size=(1,1),
                                            strides=(1,1),
                                            padding='same',
                                            name= 'unit_conv',
                                            activation=None,
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=self.l2(weight_decay))

        self.bn2 = tf.keras.layers.BatchNormalization(momentum=momentum,epsilon=epsilon)
        self.conv2 = tf.keras.layers.Conv2D(filters=k,
                                            use_bias=False,
                                            kernel_size=(3,3),
                                            strides = (1,1),
                                            padding='same',
                                            activation=None,
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=self.l2(weight_decay))

    def call(self,inputs,training):
        net = self.bn1(inputs,training=training)
        net =tf.nn.relu(net)
        net = self.conv1(net)
        net = self.bn2(net,training=training)
        net = tf.nn.relu(net)
        net = self.conv2(net)
        return net



class DenseBlock(tf.keras.Model):
    def __init__(self,k,
                 weight_decay,
                 number_of_units,
                 momentum = 0.99,
                 epsilon=0.001):
        super(DenseBlock,self).__init__()
        self.number_of_units = number_of_units
        self.units = self._add_cells([DenseUnit(k,
                                                weight_decay=weight_decay,
                                                momentum=momentum,
                                                epsilon=epsilon)
                                     for _ in range(number_of_units)])

    def _add_cells(self,cells):
        for i,c in enumerate(cells):
            setattr(self,'cell-%d'%i,c)
        return cells

    def call(self,inputs,training):
        x = self.units[0](inputs,training=training)
        for i in range(1,int(self.number_of_units)):
            output = self.units[i](x,training=training)
            x = tf.concat([x,output],axis =3)
        return x


class DenseNet(tf.keras.Model):
    def __init__(self,k,
                 units_per_block,
                 weight_decay=1e-4,
                 num_outputs=1,
                 theta=1.0,
                 momentum=0.99,
                 epsilon=0.001,
                 initial_pool=True):
        super(DenseNet,self).__init__()

        self.initial_pool = initial_pool
        self.l2 = tf.keras.regularizers.l2
        self.conv1 = tf.keras.layers.Conv2D(filters=2 * k,kernel_size = (7,7),
                                            strides = (2,2),
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=self.l2(weight_decay))
        if self.initial_pool:
            self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                                                      strides=(2,2),
                                                      padding='same')
        self.number_of_blocks = len(units_per_block)
        self.dense_blocks = self._add_cells([DenseBlock(k=k,
                                                         number_of_units=units_per_block[i],
                                                         weight_decay=weight_decay,
                                                         momentum=momentum,
                                                         epsilon=epsilon)
                                              for i in range(self.number_of_blocks)])

        self.transition_layers = self._add_cells([TransitionLayer(theta=theta,
                                                                   depth=k * units_per_block[i],
                                                                   weight_decay=weight_decay,
                                                                   momentum=momentum,
                                                                   epsilon=epsilon)
                                                   for i in range(self.number_of_blocks -1)])

        self.glo_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units = num_outputs)

    def _add_cells(self,cells):
        for i,c in enumerate(cells):
            setattr(self,'cell-%d'%i,c)
        return cells

    def call(self,inputs,training):

        net = self.conv1(inputs)
        if self.initial_pool:
            net = self.pool1(net)

        for block,transition in zip(self.dense_blocks[:-1],self.transition_layers):
            net = block(net,training=training)
            net = transition(net,training=training)

        net = self.dense_blocks[-1](net,training=training)
        net = self.glo_avg_pool(net)
        output = self.fc(net)
        output = tf.nn.sigmoid(output)
        return output

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
            write_summary,lr_schedule,early_stopping=None,verbose=1,train_from_scratch=False):
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

                lr_schedule.on_batch_end(epoch=i)

                # 计算 验证集
                val_loss = 0
                val_score = 0
                for (val_index, (x_val, y_val)) in enumerate(valDataset):
                    y_pred, val_batch_loss = self.compute_loss(X=x_val, y=y_val)
                    val_batch_score = self.compute_score(y_pred=y_pred, y_true=y_val)
                    val_loss += val_batch_loss
                    val_score += val_batch_score

                mean_val_score = val_score / (val_index + 1)
                mean_val_loss = val_loss / (val_index + 1)

                progressbar.on_batch_end(batch=ind, loss=train_loss,
                                         acc=train_score, val_loss=mean_val_loss,
                                         val_acc=mean_val_score, use_time=time.time() - start)
            # 日志记录
            write_summary.on_epoch_end(tensor=np.mean(loss_list), name='train_loss', step=i)
            write_summary.on_epoch_end(tensor=mean_val_loss, name='val_loss', step=i)
            write_summary.on_epoch_end(tensor=np.mean(score_list), name='train_acc', step=i)
            write_summary.on_epoch_end(tensor=mean_val_score, name='val_acc', step=i)

            if early_stopping:
                early_stopping.on_epoch_end(epoch=i, current=mean_val_loss)
                if early_stopping.stop_training:
                    break
            ModelCheckPoint.on_epoch_end(epoch=i, current=mean_val_loss)