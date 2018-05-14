# coding:utf-8
import numpy as np
import tensorflow as tf
import cv2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
KIND=2
def get_next_batch(batch_size=1):
    batch_x = np.zeros([batch_size*KIND, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size*KIND, KIND])
    for j in range(KIND):
        for i in range(batch_size):
            I=cv2.imread('./dataSet/'+str(j)+'/'+str(i)+'.bmp',cv2.IMREAD_GRAYSCALE)
            I=cv2.resize(I,(IMAGE_HEIGHT ,IMAGE_WIDTH))
            I=np.reshape(I,(IMAGE_HEIGHT * IMAGE_WIDTH))
            batch_x[i*KIND+j,:]=I
            vec=np.zeros((1,KIND))
            vec[0,j]=1
            batch_y[i*KIND+j,:]=vec
    return batch_x, batch_y



X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, KIND])
keep_prob = tf.placeholder(tf.float32)


def net(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    w_d = tf.Variable(w_alpha * tf.random_normal([int(IMAGE_HEIGHT * IMAGE_WIDTH) , 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024,KIND]))
    b_out = tf.Variable(b_alpha * tf.random_normal([KIND]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


def train():
    output = net()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(output, [-1, KIND])
    max_idx_p = tf.argmax(predict, 1)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, KIND]), 1)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            
            batch_x, batch_y = get_next_batch(5)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(step, loss_)
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(5)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                if acc >= 0.9:
                    saver.save(sess, "./my_test_model", global_step=step)
                    break

            step += 1



def Predict():
        image=cv2.imread('./dataSet/test/4.bmp',cv2.IMREAD_GRAYSCALE)
        image=cv2.resize(image,(IMAGE_HEIGHT ,IMAGE_WIDTH))
        image=np.reshape(image,(IMAGE_HEIGHT * IMAGE_WIDTH))
        output = net()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('.'))
            print(tf.train.latest_checkpoint('.'))
           
                #image = image.flatten() / 255
            predict = tf.argmax(tf.reshape(output, [-1,  KIND]), 1)
            label = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
            
            print (label)

             	
				
				
#train()				
Predict()				
