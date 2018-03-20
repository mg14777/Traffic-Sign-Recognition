############################################################
#                                                          #
#  Code for replicating a Shallow network with combined    #
#  pooling for fast traffic sign recognition (Zhang et al  #
#                                                          #
############################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import os
import os.path

import tensorflow as tf
import numpy as np
import batch_generator as bg
import cPickle as pickle
import time
import matplotlib.image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data-dir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max-steps', 17686,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-epochs', 45,
                            'Number of epochs. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log-frequency', 10,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model', 1000,
                            'Number of steps between model saves (default: %(default)d)')
tf.app.flags.DEFINE_integer('flush-frequency', 50,
                            'Number of steps between flushing summary results. (default: %(default)d)')
tf.app.flags.DEFINE_integer('test-log-frequency', 5,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('test-flush-frequency', 25,
                            'Number of steps between flushing summary results. (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch-size', 100, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_integer('dataset-size', 39000, 'Size of training dataset (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 1e-2, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_float('weight-decay', 0.0001, 'Weight decay (default: %(default)d)')
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum (default: %(default)d)')

# Use of Multi scale features 
tf.app.flags.DEFINE_boolean ('use_multi_scale_features',    False,     'Boolean variable indicating use of multi scale features')

# Use of dropout
tf.app.flags.DEFINE_boolean ('use_dropout',              False,     'Boolean variable indicating use of dropout')
tf.app.flags.DEFINE_float('dropout-rate', 0.7, 'Dropout rate according to AlexNet. (default: %(default)d')

# IMPORTANT: Preprocessing is same as whitening
tf.app.flags.DEFINE_boolean ('apply_whitening',              True,     'Boolean variable indicating use of whitening')

# Local Response normalization parameters (for comparison with pre processing)
tf.app.flags.DEFINE_boolean ('apply_lrn',  False,     'Add local response normalization to the network.')
tf.app.flags.DEFINE_integer ('lrn_depth_radius',     2,         'Adjacent kernel maps to consider.')
tf.app.flags.DEFINE_float   ('lrn_bias',             2,         'Bias added to LRN denominator.')
tf.app.flags.DEFINE_float   ('lrn_alpha',            10e-4,     'LRN scale parameter.')
tf.app.flags.DEFINE_float   ('lrn_beta',             0.75,      'Power to raise LRN denominator to.')

# Data augmentation
tf.app.flags.DEFINE_boolean ('augment_data',  False,     'Perform data augmentation')
tf.app.flags.DEFINE_float('rotation_angle', 0.34, 'Rotation angle for data augmentation (default: %(default)d)')


tf.app.flags.DEFINE_integer('img-width', 32, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-height', 32, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-channels', 3, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-classes', 43, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs_final/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')
# Visualization
tf.app.flags.DEFINE_boolean ('print_filters',  True, 'Flag to indicate printing filters')

# Note: Preprocessing is same as whitening 
run_log_dir = os.path.join(FLAGS.log_dir,
                           'exp_epochs_{ep}_dropout_{use_dr}_dropoutrate_{dr}_lrn_{apply_lrn}_preprocess_{apply_whitening}_multiscale_{use_multi_scale}_aug_{augment_data}'.format(ep=FLAGS.num_epochs,use_dr=FLAGS.use_dropout,dr=FLAGS.dropout_rate,apply_lrn=FLAGS.apply_lrn, apply_whitening=FLAGS.apply_whitening, use_multi_scale=FLAGS.use_multi_scale_features, augment_data=FLAGS.augment_data))
    
channel_mean = [0, 0, 0]
channel_std = [0,0,0]

def whiten_train_dataset(img):
    img = np.array(img)
    for i in range(3):
	channel_mean[i] = np.mean(img[:,0][:][:][i])
	channel_std[i] = np.std(img[:,0][:][:][i])
	img[:,0][:][:][i] = (img[:,0][:][:][i] - channel_mean[i])/channel_std[i]
    return img

def mean_removal(img):
    for j in range(0,3) :
        img_mean = np.mean(img[0][:,:,j])
        img_std = np.std(img[0][:,:,j])
        img[0][:,:,j] = (img[0][:,:,j]  - img_mean) / img_std
	return img

def whiten_test_dataset(data):
    print("normalizing test images")
    data = np.array(data)
    for i in range(0,3) :
        data[:,0][:][:][i] = (data[:,0][:][:][i]  - channel_mean[i]) / channel_std[i]
    return data

def deepnn(x_image, is_train):
    """deepnn builds the graph for a deep net for classifying TSR images.
  Args:
      x_image: an input tensor with the dimensions (N_examples, 3072), where 3072 is the
        number of pixels in a processed version of GTSRB dataset 
      is_train: boolean to indicate if the current phase is training or testing
    Returns:
      y: is a tensor of shape (N_examples, 43), with values
        equal to the logits of classifying the object images into one of 43 classes
      img_summary: a string tensor containing sampled input images.
    """

    #x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])

    img_summary = tf.summary.image('Input_images', x_image)
    

    # First convolutional layer - maps one image to 32 feature maps.
    with tf.variable_scope('Conv_1'):
		
        W_conv1 = weight_variable([5, 5, FLAGS.img_channels, 32])
	tf.add_to_collection('weights_to_decay',W_conv1)
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	
	if FLAGS.apply_lrn:
        	h_conv1 = tf.nn.lrn(h_conv1, depth_radius=FLAGS.lrn_depth_radius, bias=FLAGS.lrn_bias, alpha=FLAGS.lrn_alpha, beta=FLAGS.lrn_beta)

        if FLAGS.use_dropout:
                h_conv1 = tf.cond(is_train,lambda: tf.nn.dropout(h_conv1, keep_prob = FLAGS.dropout_rate), lambda: h_conv1)

        # Pooling layer - downsamples by 2X.
        h_pool1 = avg_pool_3x3(h_conv1)
	h_pool1_multi =  tf.nn.max_pool(h_pool1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME', name='multi_scale')

    
    with tf.variable_scope('Conv_2'):
        # Second convolutional layer -- maps 32 feature maps to another 32.
        W_conv2 = weight_variable([5, 5, 32, 32])

        tf.add_to_collection('weights_to_decay',W_conv2)
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	
        if FLAGS.apply_lrn:
                h_conv2 = tf.nn.lrn(h_conv2, depth_radius=FLAGS.lrn_depth_radius, bias=FLAGS.lrn_bias, alpha=FLAGS.lrn_alpha, beta=FLAGS.lrn_beta)


        # Second pooling layer.
        h_pool2 = avg_pool_3x3(h_conv2)
        h_pool2_multi =  tf.nn.avg_pool(h_pool2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name='multi_scale')
        if FLAGS.use_dropout:
                h_pool2 = tf.cond(is_train,lambda: tf.nn.dropout(h_pool2, keep_prob = (FLAGS.dropout_rate - 0.1)), lambda: h_pool2)

    with tf.variable_scope('Conv_3'):
        # Third convolutional layer -- maps 32 feature maps to 64.
        W_conv3 = weight_variable([5, 5, 32, 64])
        tf.add_to_collection('weights_to_decay',W_conv3)
	b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

        if FLAGS.apply_lrn:
                h_conv3 = tf.nn.lrn(h_conv3, depth_radius=FLAGS.lrn_depth_radius, bias=FLAGS.lrn_bias, alpha=FLAGS.lrn_alpha, beta=FLAGS.lrn_beta)

        # Third pooling layer.
        h_pool3 = max_pool_3x3(h_conv3)
        h_pool3_multi =  tf.nn.max_pool(h_pool3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='multi_scale')
        if FLAGS.use_dropout:
                h_pool3 = tf.cond(is_train,lambda: tf.nn.dropout(h_pool3, keep_prob = (FLAGS.dropout_rate - 0.2)), lambda: h_pool3)


    with tf.variable_scope('Conv_4'):
        # Fourth convolutional layer which is also a fully connected layer -- maps 64 feature maps to 64.
        # This is also a fully connected layer
        W_conv4 = weight_variable([4, 4, 64, 64])
        tf.add_to_collection('weights_to_decay',W_conv4)        
	b_conv4 = bias_variable([64])
        h_conv4 = tf.nn.relu(conv2d_no_padding(h_pool3, W_conv4) + b_conv4)

        if FLAGS.apply_lrn:
                h_conv4 = tf.nn.lrn(h_conv4, depth_radius=FLAGS.lrn_depth_radius, bias=FLAGS.lrn_bias, alpha=FLAGS.lrn_alpha, beta=FLAGS.lrn_beta)

        if FLAGS.use_dropout:
                h_conv4 = tf.cond(is_train,lambda: tf.nn.dropout(h_conv4, keep_prob = (FLAGS.dropout_rate - 0.3)), lambda: h_conv4)


    with tf.variable_scope('FC_2'):
	# Second fully connected layer (Conv_4 is the first fully connected layer)
        h_conv4_flat = tf.reshape(h_conv4, [-1, 1*1*64])

    	h_pool1_multi_flat = tf.contrib.layers.flatten(h_pool1_multi)
    	h_pool2_multi_flat = tf.contrib.layers.flatten(h_pool2_multi)
    	h_pool3_multi_flat = tf.contrib.layers.flatten(h_pool3_multi)
	
	if FLAGS.use_multi_scale_features:
		h_conv4_flat = tf.concat([h_pool1_multi_flat, h_pool2_multi_flat, h_pool3_multi_flat, h_conv4_flat], axis=1)
        
		W_fc = weight_variable([576, FLAGS.num_classes])
        else:
                W_fc = weight_variable([64, FLAGS.num_classes])

	tf.add_to_collection('weights_to_decay',W_fc)
        b_fc = bias_variable([FLAGS.num_classes])
        y_conv = tf.matmul(h_conv4_flat, W_fc) + b_fc
        return y_conv, img_summary
	

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='convolution')

def conv2d_no_padding(x, W):
    """conv2d_no_padding returns a 2d convolution layer with full stride but no padding"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name='convolution')

def max_pool_3x3(x):
    """max_pool_3x3 downsamples a feature map by 3X."""
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='max_pooling')

def avg_pool_3x3(x):
    """avg_pool_3x3 performs average pooling and downsamples a feature map by 3x. """
    return tf.nn.avg_pool(x,ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='avg_pooling')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial_weights = tf.random_uniform_initializer(-0.05, 0.05)
    return tf.get_variable('weights', shape, initializer=initial_weights)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial_biases = tf.random_uniform_initializer(-0.05, 0.05)
    return tf.get_variable('biases', shape,  initializer=initial_biases)

def main(_):
    tf.reset_default_graph()

    data = pickle.load(open('dataset.pkl', 'rb'))

    with tf.variable_scope('inputs'):
        # Define x_image placeholder
        x_image = tf.placeholder(tf.float32, shape=[None, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
        # Create the model
        x = tf.reshape(x_image, [-1, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
	if FLAGS.augment_data:
		rotated_images = tf.map_fn(lambda img: tf.contrib.image.rotate(img, tf.random_uniform([],-1*FLAGS.rotation_angle,FLAGS.rotation_angle)), x_image)
	        x_image = tf.concat([x_image, rotated_images], axis=0)
        # Boolean variable to distinguish training from testing
	is_train = tf.placeholder(tf.bool)
    # Build the graph for the deep net
    y_conv, img_summary = deepnn(x_image, is_train)

    # Weight decay
    with tf.variable_scope('weights_norm') as scope:
        weights_norm = tf.reduce_sum(
        input_tensor = FLAGS.weight_decay*tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection('weights_to_decay')]), name='weights_norm')

    tf.add_to_collection('losses', weights_norm)

    with tf.variable_scope('x_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    tf.add_to_collection('losses', cross_entropy)

    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    
    global_step = tf.Variable(0, trainable=False)  # this will be incremented automatically by tensorflow

    # Learning rate decay
    #decay_steps = 3000  # decay the learning rate every 3k steps
    #decay_rate  = 0.8  # the base of our exponential for the decay
    #decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, decay_rate)

    train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    incorrect_prediction = tf.not_equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    error = tf.reduce_mean(tf.cast(incorrect_prediction, tf.float32), name='error')
    
    # Summaries
    loss_summary = tf.summary.scalar('Loss', loss)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)
    error_summary = tf.summary.scalar('Error', error)
    # summaries for TensorBoard visualisation
    validation_summary = tf.summary.merge([img_summary, acc_summary, error_summary, loss_summary])
    training_summary = tf.summary.merge([img_summary, loss_summary])
    validation_error_summary = tf.summary.merge([img_summary, error_summary])
    test_summary = tf.summary.merge([img_summary, acc_summary, error_summary, loss_summary])

    with tf.variable_scope('Conv_1'):
	tf.get_variable_scope().reuse_variables()
        weights = tf.get_variable('weights')
	# Scale image values to [0,1) followed by conversion to [0,255] for display
	x_min = tf.reduce_min(weights)
	x_max = tf.reduce_max(weights)
	weights_01 = (weights - x_min)/(x_max - x_min)
	weights_uint8 = tf.image.convert_image_dtype(weights_01,dtype=tf.uint8)

        # Need to transpose the weights for tf.image_summary format [no_of_filters, height, width, channels]
	transposed_weights = tf.transpose(weights_uint8,[3,0,1,2])
        conv1_filters = tf.summary.image('First Conv Layer filters', transposed_weights, max_outputs=32)

    with tf.variable_scope('Conv_2'):
        tf.get_variable_scope().reuse_variables()
        weights = tf.get_variable('weights')
        # Average over the 32 dimensions
        weights = tf.reduce_mean(weights, 2, keep_dims=True)
        # Scale image values to [0,1) followed by conversion to [0,255] for display
        x_min = tf.reduce_min(weights)
        x_max = tf.reduce_max(weights)
        weights_01 = (weights - x_min)/(x_max - x_min)
        weights_uint8 = tf.image.convert_image_dtype(weights_01,dtype=tf.uint8)

        # Need to transpose the weights for tf.image_summary format [no_of_filters, height, width, channels]
        transposed_weights = tf.transpose(weights_uint8,[3,0,1,2])
        conv2_filters = tf.summary.image('Second Conv Layer filters', transposed_weights, max_outputs=32)
    filter_summary = tf.summary.merge([conv1_filters, conv2_filters])

    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph)
        summary_writer_test = tf.summary.FileWriter(run_log_dir + '_test', sess.graph)
	sess.run(tf.global_variables_initializer())

        # Training and validation
	
	# Calculate parameters for dataset whitening
	if FLAGS.apply_whitening:
            print('Calculating whitening parameters......')
	    '''
	    Entire training data is index 0 and testing data is index 1
	    '''
	    # Whiten train data
    	    train_data = map(lambda img: mean_removal(img), data[0])
	    train_data = whiten_train_dataset(train_data)
	    # Whiten test data
    	    test_data = map(lambda img: mean_removal(img), data[1])
	    test_data = whiten_test_dataset(test_data)
	else:
	    print('Whitening disactivated')
	    train_data = data[0]
	    test_data = data[1]
	############

        step = 0
        sys.stdout.flush()  
        evaluated_images = 0
        cumulated_error = 0
        batches_per_epoch = FLAGS.dataset_size/FLAGS.batch_size
        batch_count = 0
        for epoch in range(FLAGS.num_epochs):
            for (trainImages, trainLabels) in bg.batch_generator(train_data,'train'):
                
		(testImages, testLabels) = bg.batch_generator(test_data,'test').next()

                sys.stdout.flush()
                _, summary_str = sess.run([train_step, training_summary], feed_dict={x_image: trainImages, y_: trainLabels, is_train: True })

                if step % (FLAGS.log_frequency + 1)== 0:
                    summary_writer.add_summary(summary_str, step)
		
        ## Validation Accuracy Monitoring (Uncomment to print information)
        '''
                # Validation: Monitoring accuracy using validation set
                if step % FLAGS.log_frequency == 0:
                    validation_error, validation_accuracy, summary_str = sess.run([error, accuracy, validation_summary], feed_dict={x_image: testImages, y_: testLabels, is_train: False})

                    print('step %d, error on validation batch: %g' % (step, validation_error))
                    print('step %d, accuracy on validation batch: %g' % (step, validation_accuracy))
                    print()
		    cumulated_error = cumulated_error + validation_error
                    batch_count = batch_count + 1
                    summary_writer_validation.add_summary(summary_str, step)

                    epochs_so_far = batch_count/batches_per_epoch
		    avg_error = cumulated_error/batch_count
                    summary_writer_validation.add_summary( tf.Summary(value=[ tf.Summary.Value(tag="Average Error across batches", simple_value=avg_error),]) ,epochs_so_far)
		'''
                # Print filters learnt
	        if FLAGS.print_filters:
        	    summary_str = sess.run(filter_summary)
		    summary_writer.add_summary(summary_str,step)		    

		# Save the model checkpoint periodically.
                if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(run_log_dir + '_train', 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

                if step % FLAGS.flush_frequency == 0:
                    summary_writer.flush()
                    summary_writer_validation.flush()
                step = step + 1
	    
	    # Performance on validation set (Test set in this case)
	    validation_error_avg = 0
	    validation_loss_avg = 0
	    validation_acc_avg = 0
	    batch_count = 0
	    for (testImages,testLabels) in bg.batch_generator(test_data,'test'):
                validation_loss, validation_error, validation_accuracy, summary_str = sess.run([loss, error, accuracy, validation_summary], feed_dict={x_image: testImages, y_: testLabels, is_train: False})
		validation_error_avg += validation_error
		validation_loss_avg += validation_loss
		validation_acc_avg += validation_accuracy
		batch_count = batch_count + 1
	    validation_error_avg /= batch_count
            validation_loss_avg /= batch_count
            validation_acc_avg /= batch_count	    
	    print('Epoch %d finished' % (epoch + 1))
            print('step %d, error on validation set: %g' % (step, validation_error))
            print('step %d, accuracy on validation set: %g' % (step, validation_accuracy))
            print()            
	    summary_writer_validation.add_summary(tf.Summary(value=[ tf.Summary.Value(tag="Validation Error", simple_value=validation_error_avg),]), epoch)
            summary_writer_validation.add_summary(tf.Summary(value=[ tf.Summary.Value(tag="Validation Loss", simple_value=validation_loss_avg),]), epoch)
            summary_writer_validation.add_summary(tf.Summary(value=[ tf.Summary.Value(tag="Validation Accuracy", simple_value=validation_acc_avg),]), epoch)    
	    summary_writer_validation.flush()
	    #print('Epoch %d finished' % (epoch + 1))
        #print('Evaluated Images so far for training: %d' % (evaluated_images))
        print('Total steps: %d' % step)

        # Testing

        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0

	step = 0
        test_class_count = np.zeros([FLAGS.num_classes])
        test_class_correct = np.zeros([FLAGS.num_classes])
        
	testing_start = time.time()

        f = open('misclassification_details.txt', 'w')

        for (testImages, testLabels) in bg.batch_generator(test_data,'test'):
            classified_as, test_correct, test_accuracy_temp, test_summary_str = sess.run([y_conv, correct_prediction, accuracy, test_summary], feed_dict={x_image: testImages, y_: testLabels, is_train: False })
            testLabels = np.array(testLabels)

            batch_count = batch_count + 1
            

	    '''
	    classified_as is used for recording misclassified images (uncomment to save images)
	    '''
	    #testImages_arr = np.array(testImages)
	    #classified_as = np.array(classified_as)

            #save some of the misclassified images, it is based on the batch, thus images with the same index from different batches will be overwritten with the latest
            #misclassified one. Aditional data like the predicted class and the weights for all 43 classes is saved in misclassification_details.txt
            #for i in range(0, len(test_correct)):
            #    if not test_correct[i]:
            #            matplotlib.image.imsave('example_of_misclassified' + str(i) + '.png', testImages_arr[i])
            #            #y_conv_arr = np.array(y_conv)
            #            f.write("Image " + str(i) + " classified as class " + str(classified_as[i]) + "\n Most likely class: " + str(classified_as[i].argmax(axis=0)) + "\n")

            test_accuracy = test_accuracy + test_accuracy_temp
            test_class_count += np.sum(testLabels, axis=0)
            #test_correct = tf.reshape(tf.cast(test_correct, tf.float32),[FLAGS.batch_size,-1]).eval()
            #classified_correct_labels = testLabels*test_correct
            test_correct = np.array(test_correct)
            test_class_correct += np.sum(testLabels[test_correct], axis=0)

            if step % FLAGS.test_log_frequency == 0:
                summary_writer_test.add_summary(test_summary_str, step)
            if step % FLAGS.test_flush_frequency == 0:
                summary_writer_test.flush()
            step = step + 1

        testing_end = time.time()
        time_taken = ((testing_end - testing_start)*1000/len(data[1]))

        print('Recognition time: %.2f ms/frame' % time_taken)

        test_accuracy = test_accuracy / batch_count
        print('test set: accuracy on test set: %0.3f' % test_accuracy)

        classwise_accuracy = (test_class_correct / test_class_count)
        print('Classwise accuracy:')
        for class_id, accuracy in sorted(enumerate(classwise_accuracy), key=lambda c: c[1]):
                print('%d: %.4f' % (class_id, accuracy))

        print('Number of test batches: %d' % batch_count)
        #print('Evaluated Images: %d' % evaluated_images)

if __name__ == '__main__':
    tf.app.run(main=main)
