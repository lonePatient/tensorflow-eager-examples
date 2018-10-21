#encoding:utf-8
import cv2
import numpy as np
from math import pi
import tensorflow as tf
from math import ceil, floor

def z_score(x_train,x_test):
    mean = np.mean(x_train,axis =0)
    std = np.std(x_train,axis = 0)
    x_train = (x_train - mean) / (std + 1e-8)
    x_test = (x_test - mean) / (std + 1e-8)
    return x_train,x_test

def normalizer(image,label):
    '''
    需要注意的是：这里对dataset进行map操作时，
    :param image:
    :param label:
    :return:
    '''
    return [tf.to_float(image) / 255.,label]


# 重新定义大小
def tf_resize_images(X_img,image_size):
    '''

    :param X_img:
    :param image_size: 227
    :return:
    '''
    tf_img = tf.image.resize_images(X_img, (image_size, image_size),
                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf_img

# 尺寸缩放
def central_scale_images(X_img, scale):
    '''
    :param X_imgs:
    :param scale: 0.9 0.75 0.6
    :param image_size:
    :return:
    '''
    image_size =X_img.shape[0]
    batch_img = tf.expand_dims(X_img,axis=0)
    # Various settings needed for Tensorflow operation
    x1 = y1 = 0.5 - 0.5 * scale  # To scale centrally
    x2 = y2 = 0.5 + 0.5 * scale
    boxes = np.array([[y1, x1, y2, x2]], dtype=np.float32)
    crop_size = np.array([image_size, image_size], dtype=np.int32)
    box_ind = np.zeros((1), dtype=np.int32)
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(batch_img, boxes, box_ind, crop_size)

    return tf_img[0]


# 移动
def get_translate_parameters(index,image_size):
    if index == 0:  # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype=np.float32)
        size = np.array([image_size, ceil(0.8 * image_size)], dtype=np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * image_size))
        h_start = 0
        h_end = image_size
    elif index == 1:  # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype=np.float32)
        size = np.array([image_size, ceil(0.8 * image_size)], dtype=np.int32)
        w_start = int(floor((1 - 0.8) * image_size))
        w_end = image_size
        h_start = 0
        h_end = image_size
    elif index == 2:  # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * image_size), image_size], dtype=np.int32)
        w_start = 0
        w_end = image_size
        h_start = 0
        h_end = int(ceil(0.8 * image_size))
    else:  # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * image_size), image_size], dtype=np.int32)
        w_start = 0
        w_end = image_size
        h_start = int(floor((1 - 0.8) * image_size))
        h_end = image_size
    return offset, size, w_start, w_end, h_start, h_end

def translate_images(X_img,n_translation=0):
    '''
    :param X_imgs:
    :param n_translation: 0 1 2 3
    :return:
    '''

    if len(X_img.shape) == 3:
        batch_img = tf.expand_dims(X_img, axis=0)
    else:
        batch_img = X_img
    image_size = batch_img.shape
    offsets = np.zeros((1, 2), dtype=np.float32)
    X_translated = np.zeros((1, image_size[1], image_size[1], image_size[2]), dtype=np.float32)
    X_translated.fill(1.0)  # Filling background color
    base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(n_translation,image_size)
    offsets[:, :] = base_offset
    glimpses = tf.image.extract_glimpse(batch_img, size, offsets)
    X_translated[:, h_start: h_start + size[0], w_start: w_start + size[1], :] = glimpses
    return X_translated[0]

# 旋转90度
def rotate_images(X_img,k =1):
    '''
    :param X_imgs:
    :param k: 1 2 3 4
    :return:
    '''
    tf_img = tf.image.rot90(X_img, k)
    return tf_img

# 旋转任意角度
# 报错的
def rotate_angle_images(X_img, start_angle, end_angle,n_images = 1):
    '''
    在 start_angle 和 end_angle之间产生n_images张图像
    :param X_imgs:
    :param start_angle: -90
    :param end_angle: 90
    :n_images: 10
    :return:
    '''
    if n_images == 1:
        iterate_at = np.random.choice(180, 1)[0]
        degrees_angle = start_angle + iterate_at
        radian_value = degrees_angle * pi / 180  # Convert to radian
        tf_img = tf.contrib.image.rotate(X_img, radian_value,interpolation='BILINEAR')
        return tf_img
    else:
        X_rotate = []
        iterate_at = (end_angle - start_angle) / (n_images - 1)
        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            tf_img = tf.contrib.image.rotate(X_img, radian_value)

            X_rotate.extend(tf_img)

        X_rotate = np.array(X_rotate, dtype=np.float32)
        return X_rotate


# 左右翻转
def flip_left_right_images(X_img):
    tf_img = tf.image.flip_left_right(X_img)
    return tf_img
# 上下翻转
def flip_up_down_images(X_img):
    tf_img = tf.image.flip_up_down(X_img)
    return tf_img

# 水平翻转
def flip_transpose_images(X_img):
    tf_img = tf.image.transpose_image(X_img)
    return tf_img

# 增加噪音(增加黑点）
def add_salt_pepper_noise(X_img):
    # Need to produce a copy as to not modify the original image
    # row, col, _ = X_imgs_copy[0].shape
    img_numpy = X_img
    salt_vs_pepper = tf.constant(0.2)
    amount = tf.constant(0.004)
    num_salt = tf.ceil(amount * tf.to_float(tf.size(X_img)) * salt_vs_pepper)
    num_pepper = tf.ceil(amount * tf.to_float(tf.size(X_img)) * (1.0 - salt_vs_pepper))
    # 灰色图
    if img_numpy.shape[2] == 1:
        img_numpy = img_numpy[:,:,0]
        # Add Salt noise
        salt_coords = [np.random.randint(0, i - 1, int(num_salt))[0] for i in img_numpy.shape]
        # Add Pepper noise
        papper_coords = [np.random.randint(0, i - 1, int(num_pepper))[0] for i in img_numpy.shape]
        img_numpy[salt_coords[0], salt_coords[1]] = 1
        img_numpy[papper_coords[0], papper_coords[1]] = 0
        img_numpy = np.expand_dims(img_numpy,axis=2)
        tf_img = tf.convert_to_tensor(img_numpy)
    else:
        # Add Salt noise
        '''
        如何在tensor中进行切片赋值
        '''
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_numpy.shape]
        img_numpy[coords[0], coords[1], :] = 1
        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_numpy.shape]
        img_numpy[coords[0], coords[1], :] = 0
        tf_img = tf.convert_to_tensor(img_numpy)
    return tf_img

# 增加高斯噪音 Lighting Condition
def add_gaussian_noise(image):

    row, col, _ = image.numpy().shape
    # Gaussian distribution parameters
    gaussian = np.random.random((row, col, 1)).astype(np.float32)
    gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
    gaussian_img = cv2.addWeighted(image.numpy(), 0.75, 0.25 * gaussian, 0.25, 0)
    tf_img = tf.convert_to_tensor(gaussian_img)
    return tf_img

# 透视变换
def get_mask_coord(imshape):
    vertices = np.array([[(0.09 * imshape[1], 0.99 * imshape[0]),
                          (0.43 * imshape[1], 0.32 * imshape[0]),
                          (0.56 * imshape[1], 0.32 * imshape[0]),
                          (0.85 * imshape[1], 0.99 * imshape[0])]], dtype=np.int32)
    return vertices

def get_perspective_matrices(X_img):
    offset = 15
    img_size = (X_img.shape[1], X_img.shape[0])

    # Estimate the coordinates of object of interest inside the image.
    src = np.float32(get_mask_coord(X_img.shape))
    dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]]])

    perspective_matrix = cv2.getPerspectiveTransform(src, dst)
    return perspective_matrix

def perspective_transform(X_img):
    # Doing only for one type of example
    img_numpy = X_img.numpy()
    perspective_matrix = get_perspective_matrices(img_numpy)
    warped_img = cv2.warpPerspective(img_numpy, perspective_matrix,
                                     (img_numpy.shape[1], img_numpy.shape[0]),
                                     flags=cv2.INTER_LINEAR)
    tf_img = tf.convert_to_tensor(warped_img)
    return tf_img


# 随机裁剪图片
def random_crop_image(X_img, image_size,channel = 3,seed=2018):
    tf.set_random_seed(seed)
    #裁剪后图片分辨率保持image_size * image_size,3通道
    tf_img = tf.random_crop(X_img, [image_size, image_size, channel])
    return tf_img

#随机变化图片亮度
def random_brightness_image(X_img, seed=2018,max_delta=0.1):
    tf.set_random_seed(seed)
    tf_img = tf.image.random_brightness(X_img, max_delta=max_delta)
    return tf_img