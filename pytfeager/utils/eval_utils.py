#encoding:Utf-8
import tensorflow as tf

def r2(y_true, y_pred):
    mean_y_true = tf.reduce_mean(y_true)
    total_sum_squares = tf.reduce_sum((y_true-mean_y_true)**2)
    residual_sum_squares = tf.reduce_sum((y_true-y_pred)**2)
    r2_score = 1 - residual_sum_squares/total_sum_squares
    return r2_score

def categorical_accuracy(y_true,y_pred):
    model_pred = tf.argmax(y_pred, axis=1,output_type=tf.int64)
    actual_y_true = tf.argmax(y_true, axis=1, output_type=tf.int64)
    return tf.reduce_sum(tf.cast(tf.equal(model_pred, actual_y_true),dtype=tf.float32)) / float(y_pred.shape[0].value)

def binary_accuracy(y_true, y_pred):
    model_pred = tf.round(tf.cast(y_pred,tf.float32))
    return tf.reduce_mean(tf.cast(tf.equal(tf.cast(y_true,tf.float32),model_pred),tf.float32))

def precision(y_true, y_pred, weights=None):
    conf_matrix = tf.confusion_matrix(y_true, y_pred, num_classes=3)
    tp_and_fp = tf.reduce_sum(conf_matrix, axis=0)
    tp = tf.diag_part(conf_matrix)
    precision_scores = tp/(tp_and_fp)
    if weights:
        precision_score = tf.multiply(precision_scores, weights)/tf.reduce_sum(weights)
    else:
        precision_score = tf.reduce_mean(precision_scores)
    return precision_score

def recall(y_true, y_pred, weights=None):
    conf_matrix = tf.confusion_matrix(y_true, y_pred, num_classes=3)
    tp_and_fn = tf.reduce_sum(conf_matrix, axis=1)
    tp = tf.diag_part(conf_matrix)
    recall_scores = tp/(tp_and_fn)
    if weights:
        recall_score = tf.multiply(recall_scores, weights)/tf.reduce_sum(weights)
    else:
        recall_score = tf.reduce_mean(recall_scores)
    return recall_score

def roc_auc(y_true, y_pred, thresholds, get_fpr_tpr=True):
    tpr = []
    fpr = []
    for th in thresholds:
        # Compute number of true positives
        tp_cases = tf.where((tf.greater_equal(y_pred, th)) &
                            (tf.equal(y_true, 1)))
        tp = tf.size(tp_cases)
        # Compute number of true negatives
        tn_cases = tf.where((tf.less(y_pred, th)) &
                            (tf.equal(y_true, 0)))
        tn = tf.size(tn_cases)
        # Compute number of false positives
        fp_cases = tf.where((tf.greater_equal(y_pred, th)) &
                            (tf.equal(y_true, 0)))
        fp = tf.size(fp_cases)
        # Compute number of false negatives
        fn_cases = tf.where((tf.less(y_pred, th)) &
                            (tf.equal(y_true, 1)))
        fn = tf.size(fn_cases)
        # Compute True Positive Rate for this threshold
        tpr_th = tp / (tp + fn)
        # Compute the False Positive Rate for this threshold
        fpr_th = fp / (fp + tn)
        # Append to the entire True Positive Rate list
        tpr.append(tpr_th)
        # Append to the entire False Positive Rate list
        fpr.append(fpr_th)

    # Approximate area under the curve using Riemann sums and the trapezoidal rule
    auc_score = 0
    for i in range(0, len(thresholds) - 1):
        height_step = tf.abs(fpr[i + 1] - fpr[i])
        b1 = tpr[i]
        b2 = tpr[i + 1]
        step_area = height_step * (b1 + b2) / 2
        auc_score += step_area
    return auc_score, fpr, tpr


def tf_f1_score(y_true, y_pred):

    f1s = [0, 0, 0]

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
        FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights)

    f1s[2] = tf.reduce_sum(f1 * weights)

    micro, macro, weighted = f1s
    return micro, macro, weighted
