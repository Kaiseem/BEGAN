import tensorflow as tf
import ops
slim = tf.contrib.slim
def GenerativeCNN(z,repeat_num,h_num=8,hidden_num=64,kernal_size=3,use_bn=tf.constant(True)):
    if repeat_num==3:
        out_channal=1
    else:
        out_channal=3
    with tf.variable_scope("G") as vs:
        h = ops.fc(z,h_num*h_num*hidden_num)
        h=h0=tf.reshape(h,[-1,h_num,h_num,hidden_num])
        for i in range(repeat_num):
            #h = tf.layers.conv2d_transpose(h, filters=h, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
            h = ops.conv(h, kernal_size, hidden_num)
            h = ops.conv(h, kernal_size, hidden_num)
            if i < repeat_num-1:
                h = upsampling(h,h0)
        out= ops.conv(h, kernal_size, out_channal)
    variables = tf.contrib.framework.get_variables(vs)
    return out,variables
def DiscriminatorCNN(x,repeat_num,h_num=8, hidden_num=64,kernal_size=3,use_bn=tf.constant(True) ):#hidden_num={64,128}
    if repeat_num==3:
        out_channal=1
    else:
        out_channal=3
    with tf.variable_scope("D") as vs:
        h=ops.conv(x,kernal_size,hidden_num)
        for i in range(repeat_num):
            channel_num=hidden_num * (i + 1)
            h = ops.HDconv(h,  channel_num,1)
            h = ops.HDconv(h,  channel_num,3)
            #h = ops.HDconv(h,  channel_num,5) 125
            if i < repeat_num-1:
                #downsampling
                h = ops.conv(h, kernal_size, channel_num, 2)
        h=ops.fc(h,h_num*h_num)


        h = ops.fc(h,h_num*h_num*hidden_num)
        h=h0=tf.reshape(h,[-1,h_num,h_num,hidden_num])
        for i in range(repeat_num):
            h = ops.conv(h, kernal_size, hidden_num)
            h = ops.conv(h, kernal_size, hidden_num)
            if i < repeat_num-1:
                h = upsampling(h,h0)
        out= ops.conv(h, kernal_size, out_channal)

    variables = tf.contrib.framework.get_variables(vs)
    return out,variables

def upsampling11(x,scale):
    input_size = x.get_shape().as_list()
    return tf.image.resize_nearest_neighbor(x, (input_size[1] * scale, input_size[2] * scale))#tf.image.resize_nearest_neighbor(h0, (input_size[1]*scale,input_size[2]*scale))



def upsampling(x,h0):
    input_size = x.get_shape().as_list()
    out=tf.layers.conv2d_transpose(x, filters=input_size[-1], kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
    #print(out1.get_shape().as_list())
    #out2=tf.layers.conv2d_transpose(h0, filters=input_size[-1], kernel_size=3, strides=int(input_size[1]/8)*2, padding='same', activation=tf.nn.relu)
    #print(out2.get_shape().as_list())
    #out=tf.concat([out1,out2],axis=3)
    #print(out.get_shape().as_list())
    return out



'''

def GenerativeCNN(z,repeat_num,h_num=8,hidden_num=64,kernal_size=3,use_bn=tf.constant(True)):
    if repeat_num==3:
        out_channal=1
    else:
        out_channal=3
    with tf.variable_scope("G") as vs:
        h = ops.fc(z,h_num*h_num*hidden_num)

        h=tf.reshape(h,[-1,h_num,h_num,hidden_num])
        for i in range(repeat_num):
            h = ops.conv(h, kernal_size, hidden_num)
            h = ops.conv(h, kernal_size, hidden_num)
            if i < repeat_num-1:
                h = upsampling(h,2)
        out= ops.conv(h, kernal_size, out_channal)
    variables = tf.contrib.framework.get_variables(vs)
    return out,variables
def DiscriminatorCNN(x,repeat_num,h_num=8, hidden_num=64,kernal_size=3,use_bn=tf.constant(True) ):#hidden_num={64,128}
    if repeat_num==3:
        out_channal=1
    else:
        out_channal=3
    with tf.variable_scope("D") as vs:
        h=ops.conv(x,kernal_size,hidden_num)

        for i in range(repeat_num):
            channel_num=hidden_num * (i + 1)
            h = ops.conv(h, kernal_size, channel_num)
            h = ops.conv(h, kernal_size, channel_num)
            if i < repeat_num-1:
                h = ops.conv(h, kernal_size, channel_num, 2)
        h=ops.fc(h,h_num*h_num)
        h = ops.fc(h,h_num*h_num*hidden_num)
        h=tf.reshape(h,[-1,h_num,h_num,hidden_num])
        for i in range(repeat_num):
            h = ops.conv(h, kernal_size, hidden_num)
            h = ops.conv(h, kernal_size, hidden_num)
            if i < repeat_num-1:
                h = upsampling(h,2)
        out= ops.conv(h, kernal_size, out_channal)
    variables = tf.contrib.framework.get_variables(vs)
    return out,variables
def upsampling(x,scale):
    input_size = x.get_shape().as_list()
    return tf.image.resize_nearest_neighbor(x, (input_size[1] * scale, input_size[2] * scale))
def upsampling123123(x,h0,scale):
    input_size = x.get_shape().as_list()
    return tf.concat([tf.image.resize_nearest_neighbor(x, (input_size[1]*scale,input_size[2]*scale)),tf.image.resize_nearest_neighbor(h0, (input_size[1]*scale,input_size[2]*scale))],axis=3)




def GenerativeCNN(z, repeat_num, h_num=8, hidden_num=64, kernal_size=3, use_bn=tf.constant(True)):
    if repeat_num == 3:
        out_channal = 1
    else:
        out_channal = 3
    with tf.variable_scope("G") as vs:
        h = ops.fc(z, h_num * h_num * hidden_num)
        h = h0 = tf.reshape(h, [-1, h_num, h_num, hidden_num])
        for i in range(repeat_num):
            h = ops.conv(h, kernal_size, hidden_num)

            h = ops.conv(h, kernal_size, hidden_num)
            if i < repeat_num-1:
                h = upsampling(h,h0, 2)
        out= ops.conv(h, kernal_size, 3)
    variables = tf.contrib.framework.get_variables(vs)
    return out,variables

def DiscriminatorCNN(x,repeat_num,h_num=8, hidden_num=64,kernal_size=3,use_bn=tf.constant(True) ):#hidden_num={64,128}
    if repeat_num==3:
        out_channal=1
    else:
        out_channal=3
    with tf.variable_scope("D") as vs:
        h=ops.conv(x,kernal_size,hidden_num)
        for i in range(repeat_num):
            channel_num=hidden_num * (i + 1)
            h = ops.HDconv(h,  channel_num,1)
            h = ops.HDconv(h,  channel_num,2)
            h = ops.HDconv(h,  channel_num,5)
            if i < repeat_num-1:
                h = ops.conv(h, kernal_size, channel_num, 2)

        mean=ops.fc(h,h_num*h_num)
        std=0.5*ops.fc(h,h_num*h_num)
        epsilon = tf.random_normal(tf.stack([tf.shape(h)[0], h_num*h_num]))
        z = mean + tf.multiply(epsilon, tf.exp(std))

        h=ops.fc(z,h_num*h_num,activation_func=lrelu)
        h = ops.fc(h, h_num * h_num * hidden_num,activation_func=lrelu)
        h = h0 = tf.reshape(h, [-1, h_num, h_num, hidden_num])

        for i in range(repeat_num):
            h = ops.conv(h, kernal_size, hidden_num)
            h = ops.conv(h, kernal_size, hidden_num)
            if i < repeat_num-1:
                h = upsampling(h,h0, 2)
        out= ops.conv(h, kernal_size, 3)
    variables = tf.contrib.framework.get_variables(vs)
    return out,variables,mean,std
def lrelu(h,alpha=0.2):
    return tf.maximum(alpha*tf.nn.relu(h),tf.nn.relu(h))
def upsampling(x,h0,scale):
    input_size = x.get_shape().as_list()
    return tf.concat([tf.image.resize_nearest_neighbor(x, (input_size[1]*scale,input_size[2]*scale)),tf.image.resize_nearest_neighbor(h0, (input_size[1]*scale,input_size[2]*scale))],axis=3)

'''