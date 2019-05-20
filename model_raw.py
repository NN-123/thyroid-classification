import tensorflow as tf
import loader

learning_rate = 0.001
num_steps = 500
display_steps = 10

data = loader.load()
(train_images, train_labels), (test_images, test_labels) = loader.make_dataset(data)

train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

train = train.repeat().shuffle(len(train_images)).batch(10).prefetch(10)

iterator = train.make_initializable_iterator()
next_batch = iterator.get_next()

X = tf.placeholder(tf.float32, [None, 72, 112, 3])
Y = tf.placeholder(tf.float32, [None, 2])


def conv2d_relu(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def max_pool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, weights, biases):
    conv1 = conv2d_relu(x, weights['wc1'], biases['bc1'])
    conv2 = conv2d_relu(conv1, weights['wc2'], biases['bc2'])
    conv2 = max_pool2d(conv2, 2)
    conv2 = tf.nn.dropout(conv2, rate=0.25)

    conv3 = conv2d_relu(conv2, weights['wc3'], biases['bc3'])
    conv4 = conv2d_relu(conv3, weights['wc4'], biases['bc4'])
    conv4 = max_pool2d(conv4, 2)
    conv4 = tf.nn.dropout(conv4, rate=0.25)

    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, rate=0.5)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32])),

    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32])),

    'wc3': tf.Variable(tf.random_normal([3, 3, 32, 64])),

    'wc4': tf.Variable(tf.random_normal([3, 3, 64, 64])),

    'wd1': tf.Variable(tf.random_normal([18 * 28 * 64, 512])),

    'out': tf.Variable(tf.random_normal([512, 2]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),

    'bc2': tf.Variable(tf.random_normal([32])),

    'bc3': tf.Variable(tf.random_normal([64])),

    'bc4': tf.Variable(tf.random_normal([64])),

    'bd1': tf.Variable(tf.random_normal([512])),

    'out': tf.Variable(tf.random_normal([2]))
}

logits = conv_net(X, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y
))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(iterator.initializer)
    for step in range(num_steps):
        batch_x, batch_y = sess.run(next_batch)
        sess.run(train_op, feed_dict={
            X: batch_x,
            Y: batch_y
        })
        if step % display_steps == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Mini batch loss= " + \
                  "{:.4f}".format(loss) + ", Training accuracy= " + \
                  "{:.3f}".format(acc))
    print("Optimization Finished!")
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: test_images,
                                        Y: test_labels}))
