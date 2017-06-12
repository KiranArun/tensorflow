from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

# set up arguements
# server location, number of input points and max answer (last 2 must be same as training model)
tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_integer('points', 4, 'number of values to input')
tf.app.flags.DEFINE_integer('max_answer', 40, 'width of frame')
FLAGS = tf.app.flags.FLAGS


def main(_):
	
	# setup the parameters, must the same as model
	# number of input values
	vals = FLAGS.points
	# max answer, so basically the width of the frame
	max_answer = FLAGS.max_answer
	# number of different M's, biggest gradient will fit in frame
	gradients = max_answer/(vals)+1
	
	# constants to format input data
	length = max_answer
	full_length = length*vals
	
	# set host and port
	host, port = FLAGS.server.split(':')
	
	channel = implementations.insecure_channel(host, int(port))
	stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

	# define input line array to record input points
	input_line = np.array([])
	
	# ask user for 3 input points
	print('input values, the max gradient is', gradients-1)
	for i in range(vals):
		print('\nvalue',i+1,'?')
		val = int(input())
		input_line = np.append(input_line, val)
		
	# record the last value as we do not predict b
	l_val = input_line[-1]
	print(input_line)


	# format input data to 1d binary array
	def format_data(line_data):
		
		# set number of times to loop
		number_points = vals
		
		# remove b from all points
		# left with y = Mx
		first_value = line_data[0]
		training_line_data = np.subtract(line_data,first_value)
		
		# initialize input array
		input_data = np.zeros([vals,length])

		# loop for each point
		for a in range(number_points):
			
			# replace indexed value with 1 to make a binary array
			input_data[a,line_data[a]] = 1
				
		# reshape and return array
		input_data = input_data.reshape(1,full_length)
		return(input_data.astype(np.float32))
	
	data = format_data(input_line)
	#print(data, data.shape)
	
	
	request = predict_pb2.PredictRequest()
	
	request.model_spec.name = 'particle_predictor'
	request.model_spec.signature_name = 'predict_particle'
	
	# input data to server
	request.inputs['frames'].CopyFrom(tf.contrib.util.make_tensor_proto(data, shape=[1,full_length]))
	request.inputs['keep_prob'].CopyFrom(tf.contrib.util.make_tensor_proto(1.0))

	# get output from server
	result = stub.Predict(request, 10.0)  # 10 secs timeout
	
	# display results and predictions
	probs = result
	probs = np.array(result.outputs['scores'].float_val)
	learnt_ans = np.argmax(probs)+l_val	
	print('result =', result)
	print(probs)
	print('line =', input_line)
	print('prediction =', learnt_ans)

if __name__ == '__main__':
  tf.app.run()
