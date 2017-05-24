from __future__ import print_function
from grpc.beta import implementations
import tensorflow as tf
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_integer('points', '3', 'number of values to input')
tf.app.flags.DEFINE_string('max_answer', '40', 'width of frame')
FLAGS = tf.app.flags.FLAGS


def main(_):
	
	# setup the parameters, must the same as model
	# number of input values
	vals = FLAGS.points
	# max answer, so basically the width of the frame
	max_answer = FLAGS.max_answer
	# number of different M's, biggest gradient will fit in frame
	gradients = max_answer/(vals)+1
	
	# extra variables to format input data
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

	# function to format the data
	def test_model():
		
		test_line = input_line
		f_value = test_line[0]
		test_line = np.subtract(test_line,f_value)
		
		input_array = np.array([])
		for a in range(vals):
			test_array = np.zeros([1,length])
			test_array[0,test_line[a]] = 1
			input_array = np.append(input_array, test_array).reshape([1,(a+1)*length])
		return(input_array.astype(np.float32))
  
	data = test_model()
	print(data, data.shape)
	
	
	request = predict_pb2.PredictRequest()
		
	request.model_spec.name = 'particle_predictor'
	request.model_spec.signature_name = 'predict_particle'
	request.inputs['frames'].CopyFrom(tf.contrib.util.make_tensor_proto(data, shape=[1,full_length]))
		
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
