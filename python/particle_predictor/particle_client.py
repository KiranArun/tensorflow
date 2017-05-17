# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS


def main(_):
	
	gradients = 10
	# number of input values
	vals = 3
	
	length = (gradients-1)*vals-1

	# the full length is the length of all the input frames stacked into one, 1d array
	full_length = length*vals
	
	host, port = FLAGS.server.split(':')
  
	channel = implementations.insecure_channel(host, int(port))
  
	stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
	# Send request
  
	test_line = [1,2,3]
	def test_model():
		# first value of the points to take away the b later
		f_value = test_line[0]
    
		# take away the first value so its only left with the difference between
		# the points and predicts y = Mx , not y = Mx+ b
		for z in range(vals):
			test_line[z] = test_line[z] - f_value

		input_array = np.array([])
		# for each input value, write to the input array
		# with a 1 in the position
		for a in range(vals):
			test_array = np.zeros([1,length])
			test_array[0][test_line[a]] = 1
			# we write to array and shape so new line for each test value
			input_array = np.append(input_array, test_array).reshape([1,(a+1)*length])
        
		# return array
		return(input_array)
  
	
	# See prediction_service.proto for gRPC request/response details.
	data = test_model()
	print(data)
	request = predict_pb2.PredictRequest()
		
	request.model_spec.name = 'particle_predictor'
		
	request.model_spec.signature_name = 'predict_particle'
		
	request.inputs['frames'].CopyFrom(tf.contrib.util.make_tensor_proto(data, shape=[1,data[0].size]))
		
	result = stub.Predict(request, 10.0)  # 10 secs timeout
		
	print(result)


if __name__ == '__main__':
  tf.app.run()
