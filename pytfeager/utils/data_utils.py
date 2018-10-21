#encoding:utf-8
import os
import numpy as np
from imutils import paths
import progressbar
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pytfeager.preprocessing.data_preocessing import normalizer

def _convert_image(img_path, img_label,is_train=True):
    with tf.gfile.FastGFile(img_path, 'rb') as fid:
            image_str = fid.read()
    if is_train:
        feature_key_value_pair = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_str])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_label]))
        }
    else:
        feature_key_value_pair = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_str])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[-1]))
        }
    feature = tf.train.Features(feature=feature_key_value_pair)
    example = tf.train.Example(features=feature)
    return example

def convert_image_record(img_paths,img_labels, tfrecord_file_name,is_train=True):

    with tf.python_io.TFRecordWriter(tfrecord_file_name) as tfwriter:
        widgets = ['[INFO] write image to tfrecord: ', progressbar.Percentage(), " ",
                       progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(img_paths), widgets=widgets).start()
        for i, (img_path,img_label) in enumerate(zip(img_paths,img_labels)):
            example = _convert_image(img_path,img_label, is_train=is_train)
            tfwriter.write(example.SerializeToString())
            pbar.update(i)
        pbar.finish()

def tf_record_parser(tfrecord,image_size=(227,227)):
    features = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
    }
    sample = tf.parse_single_example(tfrecord, features)
    image = tf.image.decode_jpeg(sample['image'])
    image = tf.image.resize_images(image, image_size, method=1)
    label = sample['label']
    return [image, label]

def train_val_split(image_path,test_size,random_state):
    trainPaths = list(paths.list_images(image_path))
    trainLabels = [p.split(os.path.sep)[-1].split(".")[0] for p in trainPaths]

    split = train_test_split(trainPaths, trainLabels,
                             test_size=test_size, stratify=trainLabels,
                             random_state=random_state)
    (trainPaths, valPaths, trainLabels, valLabels) = split
    return (trainPaths,trainLabels),(valPaths,valLabels)


def get_dataset(train_tfrecord,val_tfrecord,buffer_size,batch_size):

    train_dataset = tf.data.TFRecordDataset(filenames=[train_tfrecord])
    train_dataset = train_dataset.map(tf_record_parser)
    train_dataset = train_dataset.map(normalizer)
    train_dataset = train_dataset.shuffle(buffer_size,reshuffle_each_iteration=True).batch(batch_size,drop_remainder=False)

    val_dataset = tf.data.TFRecordDataset(filenames=[val_tfrecord])
    val_dataset = val_dataset.map(tf_record_parser)
    val_dataset = val_dataset.map(normalizer)
    val_dataset = val_dataset.batch(batch_size*10,drop_remainder=False)
    return train_dataset,val_dataset

class TTA_Model():

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        pred = []
        for x_i in X:
            p0 = self.model(self._expand(x_i[:, :, 0]))
            p1 = self.model(self._expand(np.fliplr(x_i[:, :, 0])))
            p2 = self.model(self._expand(np.flipud(x_i[:, :, 0])))
            p3 = self.model(self._expand(np.fliplr(np.flipud(x_i[:, :, 0]))))
            p = (p0 +
                 self._expand(np.fliplr(p1[0][:, :, 0])) +
                 self._expand(np.flipud(p2[0][:, :, 0])) +
                 self._expand(np.fliplr(np.flipud(p3[0][:, :, 0])))
                 ) / 4
            pred.append(p)
        return np.array(pred)

    def _expand(self, x):
        return np.expand_dims(np.expand_dims(x, axis=0), axis=3)