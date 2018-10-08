import tensorflow as tf
import BEGAN_model as model
import os,math
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('training_epoch',30, "training epoch")
tf.app.flags.DEFINE_integer('training_iteration',int(162770/16), "training epoch")
tf.app.flags.DEFINE_integer('batch_size', 128, "batch size")
tf.app.flags.DEFINE_integer('test_size', 100, "batch size")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate")
tf.app.flags.DEFINE_string('save_name', 'G:/GAN/BEGAN/saved/model.ckpt', "saved parameters")
tf.app.flags.DEFINE_string('sum_name', 'G:/GAN/BEGAN/saved/summary', "saved summary")
tf.app.flags.DEFINE_boolean('is_continue_train',1, "True for continue train, False for new train")
def denorm(input):
    return (input+1)*127.7

gamma=0.5
lambda_k=0.001
def train():
    # model
    x,lenl=load_train()
    #x=load_mnist()
    train_data=x/127.5-1
    repeat_num=int(math.log(x.get_shape().as_list()[1],2))-2
    random_z=tf.random_uniform((tf.shape(train_data)[0], 64), minval=-1.0, maxval=1.0)
    G_out,G_var=model.GenerativeCNN(random_z,repeat_num)
    D_out,D_var=model.DiscriminatorCNN(tf.concat([G_out,train_data],0),repeat_num)

    #latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * std - tf.square(mean) - tf.exp(2.0 * std))#KL

    AE_G,AE_x=tf.split(D_out,2)
    loss_real = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(denorm(AE_x)-x),axis=[1,2,3])))
    loss_fake = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(denorm(AE_G)-denorm(G_out)),axis=[1,2,3])))
    #loss_fake = tf.reduce_mean(tf.abs(denorm(AE_G)-denorm(G_out)))

    global_steps = tf.Variable(0, trainable=False)
    kt = tf.Variable(0., trainable=False, name='kt')

    lossD=loss_real-kt*loss_fake
    lossG=loss_fake

    lr_G = tf.maximum(tf.train.exponential_decay(0.00008, global_steps, 100000, 0.5, staircase=False),0.00002)
    lr_D = tf.maximum(tf.train.exponential_decay(0.00008, global_steps, 100000, 0.5, staircase=False),0.00002)
    wdloss=tf.reduce_mean(tf.get_collection('losses'))
    Gtrain = tf.train.AdamOptimizer(learning_rate=lr_G,beta1=0.9,beta2=0.999).minimize(lossG+wdloss,var_list=G_var)
    Dtrain = tf.train.AdamOptimizer(learning_rate=lr_D,beta1=0.9,beta2=0.999).minimize(lossD+wdloss,var_list=D_var,global_step=global_steps)
    balance=gamma*loss_real-loss_fake
    measure = loss_real + tf.abs(balance)
    with tf.control_dependencies([Dtrain, Gtrain]):
        kt = tf.assign(kt, tf.clip_by_value(kt + lambda_k * balance, 0, 1))


    # session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1)
    coord = tf.train.Coordinator()

    # summary
    tf.summary.image('G',denorm(G_out),16,)
    tf.summary.image('input',x,16,)
    tf.summary.scalar('balance', balance)
    tf.summary.scalar('kt',kt)
    tf.summary.scalar('measure',measure)
    tf.summary.scalar('loss/loss_real', loss_real)
    tf.summary.scalar('loss/loss_fake', loss_fake)
    tf.summary.scalar('loss/d_loss', lossD)
    tf.summary.scalar('loss/g_loss', lossG)


    with tf.Session(config=tf_config) as sess:
        sess.run(init)
        merge_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.sum_name, sess.graph)
        queue_runner = tf.train.start_queue_runners(sess, coord=coord)
        tf.train.start_queue_runners(sess=sess)
        if (FLAGS.is_continue_train):

            ckpt = tf.train.get_checkpoint_state(os.getcwd()+'/saved')
            if ckpt and ckpt.model_checkpoint_path:
                print('successfully load')
                saver.restore(sess, ckpt.model_checkpoint_path)
        for epoch in range(FLAGS.training_epoch):
            for i in range(int(lenl/16)):
                g_,d_,losG,losD,train_summary,step,ktttt,mmmm=sess.run([Gtrain,Dtrain,lossG,lossD,merge_summary,global_steps,kt,measure])
                print('Epoch:%s,  Iteration:%s,   %s/%s,  lossG:%1.5f, lossD:%1.5f, kt:%1.5f   measure:%3.2f  '%(epoch,step,  i*16,lenl,losG,losD,ktttt,mmmm))
                if i%100==0:
                    train_writer.add_summary(train_summary, step)
                if i % 5000 == 0:
                    savename=FLAGS.save_name+'_'+str(step)
                    saver.save(sess, savename)
        coord.request_stop()
        coord.join(queue_runner)
        sess.close()
        os._exit(0)
def load_mnist():
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
    mnist = read_data_sets("G:/PYTHON/MNIST_data/", one_hot=True)
    input_queue = tf.train.slice_input_producer([mnist.train.images])
    batch_size=16
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size
    queue = tf.train.shuffle_batch(
        [input_queue], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    queue=tf.reshape(queue,[-1,28,28,1])
    queue=tf.image.resize_image_with_crop_or_pad(queue,32,32)
    queue=queue*255
    return queue
def load_train():
    asd=2
    rootdir = 'G:/Celeba_Resize_64/train/'
    if asd == 1:
        rootdir = 'G:/Celeba_Resize_64/train/'
    elif asd ==2:
        rootdir = 'G:/Celeba_Crop_128/'
    elif asd == 3:
        rootdir = 'G:/cartoon_face_64/'

    filelist=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        if '.jpg' in os.path.basename(list[i]).lower():
            filelist.append(rootdir+list[i])
    from PIL import Image
    filename_queue = tf.train.string_input_producer(filelist, shuffle=False)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    with Image.open(filelist[0]) as img:
        w, h = img.size
        shape = [h, w, 3]
    image = tf.image.decode_jpeg(data, channels=3)
    image.set_shape(shape)
    batch_size=16
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size
    queue = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')
    train_data=tf.cast(queue,tf.float32)
    return train_data,len(filelist)

def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()