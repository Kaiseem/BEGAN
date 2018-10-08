import tensorflow as tf
import math
def squeeze_and_excitation(input):
    input_size=input.get_shape().as_list()
    global_pool=tf.nn.avg_pool(input,[1,input_size[1],input_size[2],1],[1,1,1,1],'SAME')
    fc1=fc(global_pool,int(input_size[-1]/16),activation_func=tf.nn.relu)
    fc2=fc(fc1,int(input_size[-1]),activation_func=tf.nn.sigmoid)
    scale=tf.expand_dims(tf.expand_dims(fc2,1),1)
    return input*scale


def conv(inputs, kernel_size, output_num,strides=1, weight_decay=0.0004,activation_function=tf.nn.elu):
    input_size = inputs.get_shape().as_list()[-1]
    conv_weights = tf.Variable(
        tf.truncated_normal([kernel_size, kernel_size, input_size, output_num], dtype=tf.float32, stddev=math.sqrt(2 / (kernel_size * kernel_size * output_num))),
        name='weights')
    conv_biases = tf.Variable(tf.constant(0.0, shape=[output_num], dtype=tf.float32), 'biases')
    conv_layer = tf.nn.conv2d(inputs, conv_weights, [1, strides, strides, 1], padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, conv_biases)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(weight_decay)(conv_weights))
    if activation_function:
        conv_layer = activation_function(conv_layer)
    return conv_layer
def dconv(inputs, kernel_size, output_num,strides=1, weight_decay=0.0004,activation_function=tf.nn.elu):
    input_size = inputs.get_shape().as_list()
    conv_weights = tf.Variable(
        tf.truncated_normal([kernel_size, kernel_size,  output_num,input_size[-1]], dtype=tf.float32, stddev=math.sqrt(2 / (kernel_size * kernel_size * output_num))),
        name='weights')

    inputs_shape = tf.shape(inputs)
    out_size=[inputs_shape[0], 2*inputs_shape[1], 2*inputs_shape[2], output_num]
    conv_biases = tf.Variable(tf.constant(0.0, shape=[output_num], dtype=tf.float32), 'biases')
    conv_layer = tf.nn.conv2d_transpose(value=inputs, filter=conv_weights,output_shape=out_size,strides=[1, strides, strides, 1], padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, conv_biases)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(weight_decay)(conv_weights))
    if activation_function:
        conv_layer = activation_function(conv_layer)
    return conv_layer
#Hybrid Dilated Convolution
def HDconv(inputs,  output_num,rate=1,weight_decay=0.0004,activation_function=tf.nn.elu):
    input_size = inputs.get_shape().as_list()[-1]
    conv_weights = tf.Variable(
        tf.truncated_normal([3, 3, input_size, output_num], dtype=tf.float32, stddev=math.sqrt(2 / (3*3 * output_num))),
        name='weights')
    conv_biases = tf.Variable(tf.constant(0.0, shape=[output_num], dtype=tf.float32), 'biases')
    conv_layer = tf.nn.atrous_conv2d(inputs, conv_weights, rate, padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, conv_biases)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(weight_decay)(conv_weights))
    if activation_function:
        conv_layer = activation_function(conv_layer)
    return conv_layer

def fc(inputs, output_size,  weight_decay=0.0004,activation_func=None,stddev=0.01,init_bias=0.0,name=None,input_size=None):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) == 4:
        fc_weights = tf.Variable(tf.truncated_normal([input_shape[1] * input_shape[2] * input_shape[3], output_size], dtype=tf.float32,
                                stddev=stddev),name='weightfc'
            )
        inputs = tf.reshape(inputs, [-1, fc_weights.get_shape().as_list()[0]])
    else:
        fc_weights = tf.Variable(tf.random_normal([input_shape[-1], output_size], dtype=tf.float32, stddev=stddev),name='weightfc')

    fc_biases = tf.Variable(tf.constant(init_bias, shape=[output_size], dtype=tf.float32),name='biasfc')
    fc_layer = tf.matmul(inputs, fc_weights)
    fc_layer = tf.nn.bias_add(fc_layer, fc_biases)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(weight_decay)(fc_weights))
    if activation_func:
        fc_layer = activation_func(fc_layer)
    return fc_layer

def batch_norm_layer(x, train_phase):
    with tf.variable_scope('BN'):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)

        axises = list(range(len(x.get_shape() ) - 1))
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def dropout(x,is_train):
    return tf.cond(is_train,lambda:tf.nn.dropout(x, 0.5),lambda: tf.nn.dropout(x,1.0))

def L2_loss(weight,decay,is_train):
    return tf.cond(is_train,lambda:tf.contrib.layers.l2_regularizer(decay)(weight),lambda:0.0)

def weight_variable(shape, std_value=0.05):
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=std_value)

    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.,dtype=tf.float32, shape=shape)
    return tf.Variable(initial)
# return the conv result
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')