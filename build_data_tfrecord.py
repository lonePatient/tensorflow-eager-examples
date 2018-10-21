#encoding:utf-8
import os
import argparse
from pytfeager.utils.data_utils import train_val_split,convert_image_record
from pytfeager.config import densenet_config as config

def main():
    # 首先对数据进行分割
    trainPaths = os.path.join(config.PATH,config.DATA_PATH)
    (trainPaths,trainLabels),(valPaths,valLabels) = train_val_split(trainPaths,test_size=args['test_size'],random_state=0)
    class_to_int = config.CLASS_DICT

    # 对标签进行转换
    trainLabels = [class_to_int[tag] for tag in trainLabels]
    valLabels = [class_to_int[tag] for tag in valLabels]

    # 数据保存路径
    feature_path = os.path.join(config.PATH,config.FEATURES_PATH)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    train_tfrecord = os.path.join(feature_path,'train.tfrecords')
    test_tfrecord = os.path.join(feature_path,'test.tfrecords')

    # 将训练数据集保存tfrecord
    convert_image_record(img_paths = trainPaths,
                                    img_labels = trainLabels,
                                    tfrecord_file_name = train_tfrecord,
                                    is_train=True)
    # 将验证数据集保存tfrecord
    convert_image_record(img_paths = valPaths,
                                    img_labels = valLabels,
                                    tfrecord_file_name = test_tfrecord,
                                    is_train=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--test_size', type=int, default=0.2, help='test data size')
    args = vars(ap.parse_args())
    main()