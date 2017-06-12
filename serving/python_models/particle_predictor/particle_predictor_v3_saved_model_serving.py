# import dependancies
import tensorflow as tf
import numpy as np
import os
import sys
# import dependancies for building the model
from tensorflow.python.util import compat
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.contrib.session_bundle import exporter

# set default dirto save model
default_work_dir = '/my-files/tmp/saved_models/pp_v3'
# set default iterations
default_iterations = 5000

# set parameters from cli
# version name, iterations, dir to save model in, gpu memory usage and size frame
tf.app.flags.DEFINE_integer('version', 1, 'version number of the model.')
tf.app.flags.DEFINE_integer('iterations', default_iterations,'number of training iterations.')
tf.app.flags.DEFINE_string('work_dir', default_work_dir, 'Working directory.')
tf.app.flags.DEFINE_float('gpu_memory', 0.3, 'GPU memory alocated to training')
tf.app.flags.DEFINE_integer('max_answer', 100, 'Width of frame')

FLAGS = tf.app.flags.FLAGS

def main(_):
	# exit if any parameters not compatible
	if FLAGS.iterations <= 0:
		print 'Please specify a positive value for training iteration.'
		sys.exit(-1)
	if FLAGS.version <= 0:
		print 'Please specify a positive value for version number.'
		sys.exit(-1)
	if FLAGS.gpu_memory <= 0 or FLAGS.gpu_memory >= 1:
		print 'Please specify a positive decimal inbetween 0 and 1'
		sys.exit(-1)
		
	# set values to use as input data
	vals = 4
	# set max answer = to frame size
	max_answer = FLAGS.max_answer
	# set number of gradient to train on
	gradients = max_answer/(vals)+1
	# set gpu memory usage option
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory)
	
	# set training parameters
	iterations = FLAGS.iterations
	learning_rate = 1e-5



	# function to create line data
	def training_data():
		
		# initialize variable and arrays
		
		M = 0                  # gradient
		rows = 1               # number of rows in output array
		X = np.arange(vals+1)  # input array, x values of line, extra value for the labels
		Y = np.array([])       # output array, y values of line
		
		# loop for each gradient
		for i in range(gradients):
			
			# append a new line to Y array and reshape to have one line per row
			Y = np.append(Y, X*M).reshape(rows,vals+1)
				
			# increase gradient and rows by 1
			rows += 1
			M+=1

		# return linedata in Y array and number of rows
		return(Y.astype(np.int32),np.size(Y,0))

	# get training line data and number of lines from training_data fucntion
	training_line_data,training_lines = training_data()

	# set constants to format our data into shape depending on parameters
	length = max_answer
	full_length = length*vals

	# define format data function
	def format_data(training_line_data):

		# set values to loop over
		number_points = vals
		number_lines = training_lines

		# initialize arrays for input data and labels
		labels = np.zeros([training_lines,1,gradients])
		input_data = np.zeros([training_lines,vals,length])
			
		# loop over each line
		for i in range(number_lines):
			# loop over each point in current line
			for a in range(number_points):
				# set indexed values from line data to 1 to make a binary array
				input_data[i,a,training_line_data[i,a]] = 1
			
			# set indexed label from last line data value to 1
			labels[i,0,training_line_data[i,1]] = 1

		# reshape each line is flattened to one row
		input_data = input_data.reshape(number_lines,1,full_length)
		# return input data and labels
		return(input_data.astype(np.float32),labels)


	
	# define functions to create weights/biases
	def weight_variable(shape):
		# will return a random array in shape specified
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		# will return a random array in shape specified
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	# define the convolutional layer
	def conv2d(x, W):
		# returns the  output of conv2d 
		# x is input tensor
		# W is the weights that sweep over x
		# strides is the size steps the filter moves in each dimension
		# padding = SAME means it will pad the outside with zeros to  kepp the same size
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	# define the pooling layer
	def max_pool_2x2(x):
		# this reduces the size of the tensor
		# ksize is the window size
		# strides is the same as above
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	# define function which creates the neural net
	def deepnn(x):
		# make initial weights and biases
		# first 2 are size of  window, then input channels and 32 is the number of output channels
		W_conv1 = weight_variable([4, 4, 1, 32])
		b_conv1 = bias_variable([32])

		# reshape x to a 4d tensor, batch size , dimensions and colour channels
		x_image = tf.reshape(x, [-1,vals,length,1])
		
		# convolve the input tensor and apply the relu function 
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		# apply max pooling to the output of the convolutional layer
		# this will  halve the size of the tensor dimensions
		h_pool1 = max_pool_2x2(h_conv1)
		
		# make initial weights and biases for second layer
		# this time with 32 input channels and 64 output channels
		W_conv2 = weight_variable([4, 4, 32, 64])
		b_conv2 = bias_variable([64])

		# we do the same as the first layer
		# this will again halve the tensor dimensions
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)

		# Now we put it through the densley connected layer
		# make the weights and biases with 1024 neurons in a 2d array
		W_fc1 = weight_variable([(vals/4) * (length/4) * 64, 1024])
		b_fc1 = bias_variable([1024])

		# flatten  the tensor and apply the weights and biases
		# and apply the relu function
		h_pool2_flat = tf.reshape(h_pool2, [-1, (vals/4)*(length/4)*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
		
		# dropout to prevent overfitting
		# it is a placeholder so we can change it for training or testing
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
		
		# In Readout layer weapply the final weights and biases to find the probabilities ofeach gradient
		W_fc2 = weight_variable([1024, gradients])
		b_fc2 = bias_variable([gradients])

		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
		
		# return the probabilities and keep prob
		return(y_conv, keep_prob)
		



	# set placeholder for x, input training data
	serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
	feature_configs = {'x': tf.FixedLenFeature(shape=[full_length], dtype=tf.float32),}
	tf_example = tf.parse_example(serialized_tf_example, feature_configs)
	x = tf.identity(tf_example['x'], name='x')

	# set placeholder for y, labels
	y_ = tf.placeholder(tf.float32, [None, gradients])

	# set y_conv and keep_prob by running the deepnn function inputting x
	y_conv, keep_prob = deepnn(x)

	# work out error with cross etntropy
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

	# set optimizer to AdamOptimizer
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	# return boolean whether on whether the prediction(s) are equal to the label(s)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

	# turns boolean into float to represent accuracy percentage
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
	
	# set values to later use for creating the signature
	values, indices = tf.nn.top_k(y_conv, gradients)
	table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in xrange(gradients)]))
	prediction_classes = table.lookup(tf.to_int64(indices))
	



	# run session
	sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
	   
	# initialize variables
	sess.run(tf.global_variables_initializer())

	# get training data from format data function
	x_data, y_data = format_data(training_line_data)
	
	# run training loop
	for i in range(iterations):

		# pick accuracy testing data for this iteration and reshape so one line per row in a 2d array
		# in this example we are using the whole data set but thats because it is so small
		x_acc_test = x_data.reshape(training_lines,-1)
		y_acc_test = y_data.reshape(training_lines,-1)
		
		# pick training data for this iteration
		current_x_data = x_data[i%training_lines]
		current_y_data = y_data[i%training_lines]
		
		# print current accuracy and error 20 time in total
		if i % (iterations/20) == 0:
			# set accuracy and error by feeding in specified data with keep prob  of 1
			train_accuracy = sess.run(accuracy, feed_dict={x:x_acc_test, y_: y_acc_test, keep_prob: 1.0})
			error = sess.run(cross_entropy, feed_dict={x:current_x_data, y_: current_y_data, keep_prob: 1.0})
			
			# print accuracy and error
			print 'step', i, 'training accuracy', train_accuracy,  '\nerror =', error
			
		# run the training step
		train_step.run(feed_dict={x: current_x_data, y_: current_y_data, keep_prob: 0.5})
	   
	# print accuracy after training gas finished
	x_acc_test = x_data.reshape(training_lines,-1)
	y_acc_test = y_data.reshape(training_lines,-1)
	print("test accuracy %g" % sess.run(accuracy, feed_dict={x: x_acc_test, y_: y_acc_test, keep_prob: 1.0}))



	# set directory to save model in
	export_path_base = FLAGS.work_dir
	# name new directory with version in work dir
	export_path = os.path.join(compat.as_bytes(export_path_base),compat.as_bytes(str(FLAGS.version)))
		  
	# print where its saving the model
	print 'Exporting trained model to', export_path
	builder = saved_model_builder.SavedModelBuilder(export_path)
	
	# Build the signature_def_map
	classification_inputs = utils.build_tensor_info(serialized_tf_example)        # input with placeholder x
	keep_prob_input = utils.build_tensor_info(keep_prob)						  # input with placeholder keep_prob
	classification_outputs_classes = utils.build_tensor_info(prediction_classes)  # output of prediction classes
	classification_outputs_scores = utils.build_tensor_info(values)               # output of prediction probabilities

	# build signature for classification
	classification_signature = signature_def_utils.build_signature_def(
		inputs={
			signature_constants.CLASSIFY_INPUTS: classification_inputs,
			signature_constants.CLASSIFY_INPUTS: keep_prob_input},
		outputs={
			signature_constants.CLASSIFY_OUTPUT_CLASSES:classification_outputs_classes,
			signature_constants.CLASSIFY_OUTPUT_SCORES:classification_outputs_scores
		  },
		  method_name=signature_constants.CLASSIFY_METHOD_NAME)

	
	tensor_info_x = utils.build_tensor_info(x)          # input data
	tensor_info_kp = utils.build_tensor_info(keep_prob) # keep prob, should always be 1 when not training
	tensor_info_y = utils.build_tensor_info(y_conv)     # prediction probabilities

	# build isgnature for prediction
	prediction_signature = signature_def_utils.build_signature_def(
		  inputs={'frames': tensor_info_x, 'keep_prob': tensor_info_kp},
		  outputs={'scores': tensor_info_y},
		  method_name=signature_constants.PREDICT_METHOD_NAME)

	legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

	# save graph with signature
	builder.add_meta_graph_and_variables(
		  sess, [tag_constants.SERVING],
		  signature_def_map={'predict_particle':
				  prediction_signature,
			  signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
				  classification_signature,},
		  legacy_init_op=legacy_init_op)

	# save model
	builder.save()

	print 'Done exporting!'

if __name__ == '__main__':
	tf.app.run()
