# import tensorflow and numpy
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


# set default dirto save model
default_work_dir = '/my-files/tmp/saved_models/pp_v2'
# set default iterations
default_iterations = 5000

# set parameters from cli
# version name, iterations, dir to save model in
tf.app.flags.DEFINE_integer('version', 1, 'version number of the model.')
tf.app.flags.DEFINE_integer('iterations', default_iterations,'number of training iterations.')
tf.app.flags.DEFINE_string('work_dir', default_work_dir, 'Working directory.')
FLAGS = tf.app.flags.FLAGS

def main(_):
	# exit if any parameters not compatible
	if FLAGS.version == None:
		print 'please input a version number [--version=x]'
		sys.exit(-1)
	if FLAGS.iterations <= 0:
		print 'Please specify a positive value for training iteration.'
		sys.exit(-1)
	if FLAGS.version <= 0:
		print 'Please specify a positive value for version number.'
		sys.exit(-1)
		

	# setup the parameters
	# number of input values
	vals = 3
	# max answer, so basically the width of the frame
	max_answer = 40
	# number of different M's, biggest gradient will fit in frame
	gradients = max_answer/(vals)+1
	iterations = FLAGS.iterations
	learning_rate = 0.3

	# I am using a GPU
	# this line limits memory usage of the GPU to 0.25 when session is created
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

	# defining function to make training data
	def training_data():
    
		n = 0
		rows = 1
		# array which we are using as our x values 
		# in equation of linear line, y = Mx
		# it includes 1 extra value as this will be used as our labal
		X = np.arange(vals+1).astype(np.int32)
		# empty array to write our training data to
		Y = np.array([])
		
		# loop so it cycles through every gradient
		for i in range(gradients):
			
			Y = np.append(Y, X*n).reshape(rows,vals+1)
				
			# increase number of rows to reshape it
			rows += 1
			# increase gradient by 1
			n+=1
		
		Y = Y.astype(np.int32)
		# return the training data
		# and number of lines to learn
		return(Y,np.size(Y,0))

	# print the training data
	print(training_data())

	
	# training_line_data = training data in numbers
	# training lines = number of different lines
	training_line_data,training_lines = training_data()

	# the length is for when we convert the numbers into a binary array
	# the array will be all zeros except one, which will be 1
	# this will be the particle in this pont in time
	# each one is like a frame in a video
	length = max_answer

	# the full length is the length of all the input frames stacked into one, 1d array
	full_length = length*vals 
		
	# this function turns the data into the arrays explained above
	def set_data():
		
		# there are two arrays, one for training data and one for labels
		# input_data  is the converted traiing data
		input_data = np.zeros([training_lines,vals,length])
		# the labels will be one-hot arrays
		labels = np.zeros([training_lines,1,gradients])
		
		# this sets the values specified in the training data to one
		for i in range(training_lines):
			# we need to set each individual input value
			for a in range(vals):
				# set the value to a 1
				input_data[i,a,training_line_data[i,a]] = 1
				
			# set the label value to a 1
			labels[i,0,training_line_data[i,vals-(vals-1)]] = 1
			
		# here, we reshape it tto the full length 1d array
		input_data = input_data.reshape(training_lines,1,full_length)
		
		# return the data and labels
		return(input_data,labels)      



	# we define the weights, biases and inputs
	
	# set placeholder for x, input training data
	serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
	feature_configs = {'x': tf.FixedLenFeature(shape=[full_length], dtype=tf.float32),}
	tf_example = tf.parse_example(serialized_tf_example, feature_configs)
	x = tf.identity(tf_example['x'], name='x'))
	
	# weights and biases
	W = tf.Variable(tf.zeros([full_length, gradients]))
	b = tf.Variable(tf.zeros([gradients]))
	
	# function which gives the probabilities
	y = tf.matmul(x, W) + b

	# we will feed the labels in here
	y_ = tf.placeholder(tf.float32, [None, gradients])

	# configure the loss function, using cross entropy
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	# define the optimizer
	# AdagradOptimizer works much better than GradientDescentOptimizer
	#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
	train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)
	
	# set values to later use for creating the signature
	values, indices = tf.nn.top_k(y_conv, gradients)
	table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in xrange(gradients)]))
	prediction_classes = table.lookup(tf.to_int64(indices))




	# create interactive session using the GPU line for above
	sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options)) 

	# initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# now we run the learning loop
	for _ in range(iterations):

		# set training data and labels
		x_data, y_data = set_data()
		
		# use next line each step
		x_data = x_data[_%training_lines]
		y_data = y_data[_%training_lines]
		
		# run the training optimizer
		sess.run(train_step, feed_dict={x: x_data, y_: y_data})

		# print steps and error 20 times in total
		if _ % (iterations/20) == 0:
			print 'step', _, 'out of', iterations
			print 'error =', sess.run(cross_entropy, feed_dict={x: x_data, y_: y_data})




	# set directory to save model in
	export_path_base = FLAGS.work_dir
	# name new directory with version in work dir
	export_path = os.path.join(compat.as_bytes(export_path_base),compat.as_bytes(str(FLAGS.version)))
		  
	# print where its saving the model
	print 'Exporting trained model to', export_path
	builder = saved_model_builder.SavedModelBuilder(export_path)
	
	# Build the signature_def_map
	classification_inputs = utils.build_tensor_info(serialized_tf_example)        # input with placeholder x
	classification_outputs_classes = utils.build_tensor_info(prediction_classes)  # output of prediction classes
	classification_outputs_scores = utils.build_tensor_info(values)               # output of prediction probabilities

	# build signature for classification
	classification_signature = signature_def_utils.build_signature_def(
		inputs={
			signature_constants.CLASSIFY_INPUTS: classification_inputs},
		outputs={
			signature_constants.CLASSIFY_OUTPUT_CLASSES:classification_outputs_classes,
			signature_constants.CLASSIFY_OUTPUT_SCORES:classification_outputs_scores
		  },
		  method_name=signature_constants.CLASSIFY_METHOD_NAME)

	
	tensor_info_x = utils.build_tensor_info(x)          # input data
	tensor_info_y = utils.build_tensor_info(y_conv)     # prediction probabilities

	# build isgnature for prediction
	prediction_signature = signature_def_utils.build_signature_def(
		  inputs={'frames': tensor_info_x},
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

	builder.save()

	print 'Done exporting!'


if __name__ == '__main__':
	tf.app.run()
