import init_set
import preproc
import numpy as np
import tensorflow as tf

if __name__ == '__main__':

    sess = init_set.init()

    data_train, result_train = preproc.get_data(True)
    poster_train = preproc.get_poster(True)
    data_test, _ = preproc.get_data(False)
    poster_test = preproc.get_poster(True)
    total_train_data = data_train.shape[0]
    total_test_data = data_test.shape[0]

    X_data = tf.placeholder(tf.float32, shape=[None, 8])
    X_poster = tf.placeholder(tf.float32, shape=[None, 300, 200, 3])
    Y = tf.placeholder(tf.float32, shpae=[None, 2])
    is_training = tf.placeholder(tf.bool)

    # CNN for poster
    CW1 = tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=0.01))
    CL1 = tf.nn.conv2d(X_poster, filter=W1, strides=[1,1,1,1], padding='SAME')
    CL1 = tf.contrib.layers.batch_norm(CL1, is_training=is_training, center=True, scale=True, updates_collections=None)
    CL1 = tf.nn.relu(CL1)
    CL1 = tf.nn.dropout(CL1, keep_prob)
    CW2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))
    CL2 = tf.nn.conv2d(CL1, filter=CW2, strides=[1,1,1,1], padding='SAME')
    CL2 = tf.contrib.layers.batch_norm(CL2, is_training=is_training, center=True, scale=True, updates_collections=None)
    CL2 = tf.nn.relu(CL2)
    CL2 = tf.nn.max_pool(CL2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    CL2 = tf.nn.dropout(CL2, keep_prob)

    CW3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    CL3 = tf.nn.conv2d(CL2, filter=CW3, strides=[1,1,1,1], padding='SAME')
    CL3 = tf.contrib.layers.batch_norm(CL3, is_training=is_training, center=True, scale=True, updates_collections=None)
    CL3 = tf.nn.relu(CL3)
    CL3 = tf.nn.dropout(CL3, keep_prob)
    CW4 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01))
    CL4 = tf.nn.conv2d(CL3, filter=CW4, strides=[1,1,1,1], padding='SAME')
    CL4 = tf.contrib.layers.batch_norm(CL4, is_training=is_training, center=True, scale=True, updates_collections=None)
    CL4 = tf.nn.relu(CL4)
    CL4 = tf.nn.max_pool(CL4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    CL4 = tf.nn.dropout(CL4, keep_prob)

    CW5 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    CL5 = tf.nn.conv2d(CL4, filter=CW5, strides=[1,1,1,1], padding='SAME')
    CL5 = tf.contrib.layers.batch_norm(CL5, is_training=is_training, center=True, scale=True, updates_collections=None)
    CL5 = tf.nn.relu(CL5)
    CL5 = tf.nn.dropout(CL5, keep_prob)
    CW6 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01))
    CL6 = tf.nn.conv2d(CL5, filter=CW6, strides=[1,1,1,1], padding='SAME')
    CL6 = tf.contrib.layers.batch_norm(CL6, is_training=is_training, center=True, scale=True, updates_collections=None)
    CL6 = tf.nn.relu(CL6)
    CL6 = tf.nn.max_pool(CL6, ksize=[1,5,5,1], strides=[1,5,5,1], padding='SAME')
    CL6 = tf.nn.dropout(CL6, keep_prob)

    CW7 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
    CL7 = tf.nn.conv2d(CL6, filter=CW7, strides=[1,1,1,1], padding='SAME')
    CL7 = tf.contrib.layers.batch_norm(CL7, is_training=is_training, center=True, scale=True, updates_collections=None)
    CL7 = tf.nn.relu(CL7)
    CL7 = tf.nn.dropout(CL7, keep_prob)
    CW8 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01))
    CL8 = tf.nn.conv2d(CL7, filter=CW8, strides=[1,1,1,1], padding='SAME')
    CL8 = tf.contrib.layers.batch_norm(CL8, is_training=is_training, center=True, scale=True, updates_collections=None)
    CL8 = tf.nn.relu(CL8)
    CL8 = tf.nn.max_pool(CL8, ksize=[1,5,5,1], strides=[1,5,5,1], padding='SAME')
    CL8 = tf.nn.dropout(CL8, keep_prob)
    CL8 = tf.reshape(CL8, [-1, 3*2*256])

    CW9 = tf.Variable(tf.random_normal([3*2*256, 128], stddev=0.01))
    CL9 = tf.matmul(CL8, CW9)
    CL9 = tf.contrib.layers.batch_norm(CL9, is_training=is_training, center=True, scale=True, updates_collections=None)
    CL9 = tf.nn.relu(CL9)
    CL9 = tf.nn.dropout(CL9, keep_prob)

    CW10 = tf.Variable(tf.random_normal([128, 1], stddev=0.01))
    CB10 = tf.Variable(tf.random_normal(shape=[1], stddev=0.01))
    poster = tf.nn.relu(tf.matmul(CL9, CW10) + CB10)

    # Combine poster and X_data
    X = np.concatenate((poster, X_data), axis=1)
    W1 = tf.Variable(tf.random_normal([9, 128], stddev=0.01))
    B1 = tf.Variable(tf.random_normal(shape=[128], stddev=0.01))
    L1 = tf.matmul(X, W1) + B1
    L1 = tf.contrib.layers.batch_norm(L1, is_training=is_training, center=True, scale=True, updates_collections=None)
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.dropout(L1, keep_prob)

    W2 = tf.Variable(tf.random_normal([128, 64], stddev=0.01))
    B2 = tf.Variable(tf.random_normal(shape=[64], stddev=0.01))
    L2 = tf.matmul(L1, W2) + B2
    L2 = tf.contrib.layers.batch_norm(L2, is_training=is_training, center=True, scale=True, updates_collections=None)
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.dropout(L2, keep_prob)

    W3 = tf.Variable(tf.random_normal([64, 32], stddev=0.01))
    B3 = tf.Variable(tf.random_normal(shape=[32], stddev=0.01))
    L3 = tf.matmul(L2, W3) + B3
    L3 = tf.contrib.layers.batch_norm(L3, is_training=is_training, center=True, scale=True, updates_collections=None)
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.dropout(L3, keep_prob)

    W4 = tf.Variable(tf.random_normal([32, 2], stddev=0.01))
    B4 = tf.Variable(tf.random_normal(shape=[2], stddev=0.01))
    L4 = tf.matmul(L3, W4) + B4
    L4 = tf.contrib.layers.batch_norm(L4, is_training=is_training, center=True, scale=True, updates_collections=None)
    model = tf.nn.sigmoid(L4)

    cost = -tf.reduce_mean(Y*tf.log(model) + (1-Y)*tf.log(1-model))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
