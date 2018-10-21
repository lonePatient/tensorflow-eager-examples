#encoding:utf-8
import tensorflow as tf

def mse_loss(y_pred,y_true):
    loss = tf.reduce_mean(tf.square(y_pred - y_true))
    return loss

# 好像会出现nan情况
def softmax_cross_entropy_loss(y_pred,y_true):
    y_true = tf.cast(y_true,tf.float32)
    loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1]))
    return loss

def sigmoid_cross_entropy_loss(y_pred,y_true):
    y_true = tf.cast(y_true,tf.float32)
    return tf.reduce_mean(-y_true * tf.log(y_pred) - (1 - y_true) * tf.log(1 - y_pred))

def reconstruction_loss(image_size,x_true,x_pred):
    loss = image_size * tf.nn.sigmoid_cross_entropy_with_logits(labels=x_true,logits=x_pred)
    return loss

def KL_divergence(log_var,mu):
    kl_div =  - 0.5 * tf.reduce_mean(1.0 + log_var - tf.square(mu) - tf.exp(log_var),axis =-1)
    return kl_div

def contrastive_loss(logits1,logits2,label,margin,eps=1e-7):
    Dw = tf.sqrt(eps + tf.reduce_sum(tf.square(logits1 - logits2),axis=1))
    loss = tf.reduce_sum((1.0 - tf.to_float(label)) * tf.square(Dw) + tf.to_float(label) * tf.square(tf.maximum(margin - Dw,0)))
    return loss,Dw