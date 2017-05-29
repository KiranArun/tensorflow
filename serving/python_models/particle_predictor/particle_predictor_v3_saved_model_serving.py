import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.python.util import compat
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.contrib.session_bundle import exporter

default_work_dir = '/my-files/tmp/saved_models/pp_v3'
default_iterations = 5000

# set parameters from cli
tf.app.flags.DEFINE_integer('version', 1, 'version number of the model.')
tf.app.flags.DEFINE_integer('iterations', default_iterations,'number of training iterations.')
tf.app.flags.DEFINE_string('work_dir', default_work_dir, 'Working directory.')
tf.app.flags.DEFINE_float('gpu_memory', 0.3, 'GPU memory alocated to training')
tf.app.flags.DEFINE_integer('max_answer', 100, 'Width of frame')

FLAGS = tf.app.flags.FLAGS

# exit if any parameters not compatible
def main(_):
	if FLAGS.iterations <= 0:
		print 'Please specify a positive value for training iteration.'
		sys.exit(-1)
	if FLAGS.version <= 0:
		print 'Please specify a positive value for version number.'
		sys.exit(-1)
	if FLAGS.gpu_memory <= 0 or FLAGS.gpu_memory >= 1:
		print 'Please specify a positive decimal inbetween 0 and 1'
		sys.exit(-1)
		

	vals = 4
	max_answer = FLAGS.max_answer
	gradients = max_answer/(vals)+1
	iterations = FLAGS.iterations
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory)
	learning_rate = 1e-5

#############################################################################################
#############################################################################################

	def training_data():
		
		M = 0
		rows = 1
		X = np.arange(vals+1)
		Y = np.array([])
		
		for i in range(gradients):
			
			Y = np.append(Y, X*M).reshape(rows,vals+1)
				
			rows += 1
			M+=1

		return(Y.astype(np.int32),np.size(Y,0))

	training_line_data,training_lines = training_data()

	length = max_answer
	full_length = length*vals

	def format_data(training_line_data):

		number_points = vals
		number_lines = training_lines

		labels = np.zeros([training_lines,1,gradients])
		input_data = np.zeros([training_lines,vals,length])
			
		for i in range(number_lines):
			for a in range(number_points):
				input_data[i,a,training_line_data[i,a]] = 1

			labels[i,0,training_line_data[i,1]] = 1

		input_data = input_data.reshape(number_lines,1,full_length)
		return(input_data.astype(np.float32),labels)
		
#############################################################################################
#############################################################################################

	def weight_variable(shape):
	  initial = tf.truncated_normal(shape, stddev=0.1)
	  return tf.Variable(initial)

	def bias_variable(shape):
	  initial = tf.constant(0.1, shape=shape)
	  return tf.Variable(initial)

	def conv2d(x, W):
	  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(x):
	  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


	def deepnn(x):
		W_conv1 = weight_variable([vals, length, 1, 32])
		b_conv1 = bias_variable([32])

		x_image = tf.reshape(x, [-1,vals,length,1])

		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)
		
		W_conv2 = weight_variable([vals, length, 32, 64])
		b_conv2 = bias_variable([64])

		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)

		W_fc1 = weight_variable([(vals/4) * (length/4) * 64, 1024])
		b_fc1 = bias_variable([1024])

		h_pool2_flat = tf.reshape(h_pool2, [-1, (vals/4)*(length/4)*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
		
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
		
		W_fc2 = weight_variable([1024, gradients])
		b_fc2 = bias_variable([gradients])

		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
		
		return(y_conv, keep_prob)
		
#############################################################################################
#############################################################################################

	serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
	feature_configs = {'x': tf.FixedLenFeature(shape=[full_length], dtype=tf.float32),}
	tf_example = tf.parse_example(serialized_tf_example, feature_configs)

	x = tf.identity(tf_example['x'], name='x')

	y_ = tf.placeholder(tf.float32, [None, gradients])

	y_conv, keep_prob = deepnn(x)

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
	
	values, indices = tf.nn.top_k(y_conv, gradients)
	table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in xrange(gradients)]))
	prediction_classes = table.lookup(tf.to_int64(indices))
	
#############################################################################################
#############################################################################################

	sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
	   
	sess.run(tf.global_variables_initializer())

	for i in range(iterations):

		x_data, y_data = format_data(training_line_data)
		x_acc_test = x_data.reshape(training_lines,-1)
		y_acc_test = y_data.reshape(training_lines,-1)
		
		x_data = x_data[i%training_lines]
		y_data = y_data[i%training_lines]
		
		if i % (iterations/20) == 0:
			train_accuracy = sess.run(accuracy, feed_dict={x:x_acc_test, y_: y_acc_test, keep_prob: 1.0})
			error = sess.run(cross_entropy, feed_dict={x:x_data, y_: y_data, keep_prob: 1.0})
			
			print 'step', i, 'training accuracy', train_accuracy,  '\nerror =', error
			
		train_step.run(feed_dict={x: x_data, y_: y_data, keep_prob: 0.5})
	   
	x_data, y_data = format_data(training_line_data)

	x_acc_test = x_data.reshape(training_lines,-1)
	y_acc_test = y_data.reshape(training_lines,-1)

	print("test accuracy %g" % sess.run(accuracy, feed_dict={x: x_acc_test, y_: y_acc_test, keep_prob: 1.0}))

#############################################################################################
#############################################################################################

	export_path_base = FLAGS.work_dir

	export_path = os.path.join(compat.as_bytes(export_path_base),compat.as_bytes(str(FLAGS.version)))
		  
	print 'Exporting trained model to', export_path
	builder = saved_model_builder.SavedModelBuilder(export_path)
	
	# Build the signature_def_map.
	classification_inputs = utils.build_tensor_info(serialized_tf_example)
	keep_prob_input = utils.build_tensor_info(keep_prob)
	classification_outputs_classes = utils.build_tensor_info(prediction_classes)
	classification_outputs_scores = utils.build_tensor_info(values)

	classification_signature = signature_def_utils.build_signature_def(
		inputs={
			signature_constants.CLASSIFY_INPUTS: classification_inputs,
			signature_constants.CLASSIFY_INPUTS: keep_prob_input},
		outputs={
			signature_constants.CLASSIFY_OUTPUT_CLASSES:classification_outputs_classes,
			signature_constants.CLASSIFY_OUTPUT_SCORES:classification_outputs_scores
		  },
		  method_name=signature_constants.CLASSIFY_METHOD_NAME)

	tensor_info_x = utils.build_tensor_info(x)
	tensor_info_kp = utils.build_tensor_info(keep_prob)
	tensor_info_y = utils.build_tensor_info(y_conv)

	prediction_signature = signature_def_utils.build_signature_def(
		  inputs={'frames': tensor_info_x, 'keep_prob': tensor_info_kp},
		  outputs={'scores': tensor_info_y},
		  method_name=signature_constants.PREDICT_METHOD_NAME)

	legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

	builder.add_meta_graph_and_variables(
		  sess, [tag_constants.SERVING],
		  signature_def_map={'predict_particle':
				  prediction_signature,
			  signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
				  classification_signature,},
		  legacy_init_op=legacy_init_op)

	builder.save()

	print 'Done exporting!'

if __name__ == '__main__':
	tf.app.run()
