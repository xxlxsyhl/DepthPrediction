from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import dataload
import utils

data_dict = None
vgg_npy_path = "weights/vgg19.npy"
DATA_SET_PATH = 'E:/NYU_Depth/labeled/nyu_depth_v2_training.mat'
is_training = True

TRAIN_STEP = 100000
BATCH_SIZE = 16

Height = 160
Width = 120

Out_Height = 80
Out_Width = 60

lr = 0.00001

def avg_pool(input, name):
    return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool(input, name):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv_filter(in_filters, out_filters):
    init = tf.truncated_normal(shape=[3, 3, in_filters, out_filters], dtype="float32")
    return tf.Variable(init)

def conv_bias(out_filters):
    init = tf.constant(0.1, dtype=tf.float32, shape=[out_filters])
    return tf.Variable(init, dtype="float32")

def get_conv_filter( name):
    init = tf.constant(data_dict[name][0])
    return tf.Variable(init, dtype="float32")


def get_conv_bias(name):
    init = tf.constant(data_dict[name][1])
    return tf.Variable(init, dtype="float32")

def conv_layer(input, name):
    filter = get_conv_filter(name)
    conv = tf.nn.conv2d(input, filter, [1, 1, 1, 1], padding='SAME')
    bias = get_conv_bias(name)
    return tf.nn.leaky_relu(tf.nn.bias_add(conv, bias))


def deconv_layer(input, in_filters, out_filters, flag):
    shape = tf.shape(input)
    filter = conv_filter(out_filters, in_filters)
    deconv = tf.nn.conv2d_transpose(input, filter, [BATCH_SIZE, shape[1]*2-flag, shape[2]*2, out_filters], [1, 2, 2, 1], padding='SAME')
    bias = conv_bias(out_filters)
    x = tf.nn.bias_add(deconv, bias)
    bn = batch_norm(x, is_training=is_training)
    return tf.nn.leaky_relu(bn)


def conv_layer2(input, in_filters, out_filters, name=None):
    filter = conv_filter(in_filters, out_filters)
    conv = tf.nn.conv2d(input, filter,[1, 1, 1, 1], padding='SAME' )
    bias = conv_bias(out_filters)
    return tf.nn.leaky_relu(tf.nn.bias_add(conv, bias))

def batch_norm(inputs, is_training,  epsilon = 0.001, decay = 0.99):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)

'''
def batch_norm(x, scope, is_training, epsilon=0.001, decay=0.99):
    """
    Returns a batch normalization layer that automatically switch between train and test phases based on the 
    tensor is_training

    Args:
        x: input tensor
        scope: scope name
        is_training: boolean tensor or variable
        epsilon: epsilon parameter - see batch_norm_layer
        decay: epsilon parameter - see batch_norm_layer

    Returns:
        The correct batch normalization layer based on the value of is_training
    """
    # assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool

    if is_training:
        bn = batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None)
    else:
        bn = batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True)
    return bn

    return tf.cond(
        is_training,
        lambda: batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
        lambda: batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
    )

def batch_norm_layer(x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        shape = x.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, [0,1,2]) # range(len(shape)-1)
            update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return output
'''

data_dict = np.load(vgg_npy_path, encoding='latin1').item()

x = tf.placeholder(dtype="float32", shape=[None,Width, Height,3], name="input")
y = tf.placeholder(dtype="float32", shape=[None,Out_Width,Out_Height], name="depth")

y_reshape = tf.reshape(y, [-1, Out_Width*Out_Height])
# convolution
# 160*120 64
conv1_1 = conv_layer(x, "conv1_1")
conv1_2 = conv_layer(conv1_1, "conv1_2")
pool1 = max_pool(conv1_2, 'pool1')
# 80*60 128
conv2_1 = conv_layer(pool1, "conv2_1")
conv2_2 = conv_layer(conv2_1, "conv2_2")
pool2 = max_pool(conv2_2, 'pool2')
# 40*30 256
conv3_1 = conv_layer(pool2, "conv3_1")
conv3_2 = conv_layer(conv3_1, "conv3_2")
conv3_3 = conv_layer(conv3_2, "conv3_3")
conv3_4 = conv_layer(conv3_3, "conv3_4")
pool3 = max_pool(conv3_4, 'pool3')
# 20*15 512
conv4_1 = conv_layer(pool3, "conv4_1")
conv4_2 = conv_layer(conv4_1, "conv4_2")
conv4_3 = conv_layer(conv4_2, "conv4_3")
conv4_4 = conv_layer(conv4_3, "conv4_4")
pool4 = max_pool(conv4_4, 'pool4')
# 10*7 512
conv5_1 = conv_layer(pool4, "conv5_1")
conv5_2 = conv_layer(conv5_1, "conv5_2")
conv5_3 = conv_layer(conv5_2, "conv5_3")
conv5_4 = conv_layer(conv5_3, "conv5_4")

# deconvolution
# 10*7 512
deconv6 = deconv_layer(conv5_4, 512, 64, 1)
conv6 = conv_layer2(deconv6, 64, 64, "conv6")
# 20*14
deconv7 = deconv_layer(conv6, 64, 8, 0)
conv7 = conv_layer2(deconv7, 8, 8, "conv7")
# 40*28
deconv8 = deconv_layer(conv7, 8, 1, 0)
conv8 = conv_layer2(deconv8, 1, 1, "conv8")
# 80*56
y_ = tf.reshape(conv8, [-1, Out_Width*Out_Height])




# loss function
loss = tf.reduce_sum (tf.square(y_-y_reshape))
train = tf.train.AdamOptimizer(lr).minimize(loss)

data = dataload.load_data(is_training)

saver = tf.train.Saver()

sess = tf.Session()


if is_training:
    # sess.run(tf.initialize_all_variables())
    saver.restore(sess, "weights/weights34790.ckpt")
else:
    saver.restore(sess,"weights/weights45080.ckpt")

loss_list = []

for step in range(TRAIN_STEP):
    images, depths = dataload.get_batch(data, BATCH_SIZE)
    # print(depths[0])
    # print(sess.run(conv8, feed_dict={x:images, y:depths}))
    # print("image:", images[8])
    if is_training:
        if step % 490 == 0:
            loss_value = sess.run(loss, feed_dict={x:images, y:depths})
            loss_list.append(loss_value)
            np.savetxt("loss.txt", loss_list)
            print(step, loss_value)

        if step % 490 == 0 and step != 0:
            y_value = sess.run(conv8, feed_dict={x: images})
            for idx in [0, 8, 15]:
                # print(y_value.shape)
                plt.imsave("images/"+str(step)+"_image.jpg", images[idx])
                plt.imsave("images/"+str(step)+"_depth.png", depths[idx])
                plt.imsave("images/"+str(step)+"_result.png", y_value[idx,:,:,0])
            saver.save(sess, "weights/weights"+str(step)+".ckpt")


        sess.run(train, feed_dict={x: images, y: depths})
    else:
        result = sess.run(conv8, feed_dict={x:images})
        result = result.reshape([BATCH_SIZE, Out_Width, Out_Height])
        plt.figure("image and depth")
        plt.subplot(221)
        plt.imshow(images[0])
        plt.imsave("rgb.jpg", images[0])

        plt.subplot(222)
        plt.imshow(result[0])
        plt.imsave("depth.png", result[0])

        plt.subplot(223)
        plt.imshow(images[0])

        plt.subplot(224)
        plt.imshow(depths[0])


        plt.show()

        print("depht", depths[0][0])
        print("result", result[0][0])

