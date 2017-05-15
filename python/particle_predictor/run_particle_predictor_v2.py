import tensorflow as tf
import numpy as np

model_path = "/tmp/saved_models/model.ckpt"
gradients = 10
vals = 3

length = (gradients-1)*vals-1

# the full length is the length of all the input frames stacked into one, 1d array
full_length = length*vals


x = tf.placeholder(tf.float32, [None, full_length])
# weights and biases
W = tf.Variable(tf.zeros([full_length, gradients]))
b = tf.Variable(tf.zeros([gradients]))
# function which gives the output of training data
y = tf.matmul(x, W) + b

saver = tf.train.Saver()
init = tf.initialize_all_variables()

with tf.Session() as sess:
	
	#sess.run(init)
	
	saver.restore(sess, model_path)
	
	# initialize array to record test values
	test_line = np.array([])

	print 'input values, the max gradient is', gradients-1
	# loop to ask user for input values
	for i in range(vals):
		print '\nvalue',i+1,'?'
		val = int(input())
		test_line = np.append(test_line, val)
    
	# record the last value of the points
	# this is needed because we have only worked out the gradient not the bias
	l_val = test_line[-1]
	# show line that user has inputed
	print(test_line)

	# function to convert data into training data format
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

	# print array
	#test_model()


	# run the function to get output
	# using new weights and biases
	in_array = test_model()
	probs = sess.run(y, feed_dict={x:in_array})
	# now we have found the predicted gradient,
	# we add the last value to get the next value
	learnt_ans = np.argmax(probs)+l_val
	# "true" answer, should work with all linear lines
	# this numerically works out hte gradient by findingthe difference,
	# then also adds the last value
	answer = (test_line[1]-test_line[0])+l_val

	if answer == learnt_ans:
		print 'Correct'
		
		# print learnt answer and it probability
		print '\nLearnt answer =', learnt_ans
		print 'Probaility of Gradient', np.argmax(probs), '=', probs[0][learnt_ans-l_val]
	else:
		print 'Wrong'
		
		# print learnt answer and it probability
		print '\nLearnt answer =', learnt_ans
		print 'Probaility of Gradient', np.argmax(probs), '=', probs[0][learnt_ans-l_val]
		
		# print true answer and its probabilities
		print '\nNumerical answer =', answer
		print 'Probability of Gradient', test_line[1]-test_line[0], '=', probs[0][answer-l_val]
		

	# print the probabilities
	print '\n\n', probs
