# import tensorflow and numpy
import tensorflow as tf
import numpy as np

# setup the parameters
# number of input values
vals = 3
# max answer, so basically the width of the frame
max_answer = 40
# number of different M's, biggest gradient will fit in frame
gradients = max_answer/(vals)+1
iterations = 11000
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


# data = training data
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
    in_data = np.zeros([training_lines,vals,length])
    # the labels will be one-hot arrays
    lab = np.zeros([training_lines,1,gradients])
    
    # this sets the values specified in the training data to one
    for i in range(training_lines):
        # we need to set each individual input value
        for a in range(vals):
            # set the value to a 1
            in_data[i,a,training_line_data[i,a]] = 1
            
        # set the label value to a 1
        lab[i,0,training_line_data[i,vals-(vals-1)]] = 1
        
    # here, we reshape it tto the full length 1d array
    in_data = in_data.reshape(training_lines,1,full_length)
    
    # return the data and labels
    return(in_data,lab)    

# print converted data
#print(set_data())


###################################################################################
###################################################################################


# we define the weights, biases and inputs
# this will be input training data
x = tf.placeholder(tf.float32, [None, full_length])
# weights and biases
W = tf.Variable(tf.zeros([full_length, gradients]))
b = tf.Variable(tf.zeros([gradients]))
# function which gives the output of training data
y = tf.matmul(x, W) + b

# we will feed the labels in here
y_ = tf.placeholder(tf.float32, [None, gradients])

# configure the loss function, using cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# define the optimizer
# AdagradOptimizer works much better than GradientDescentOptimizer
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)


###################################################################################
###################################################################################


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




###################################################################################
###################################################################################




# initialize array to record test values
test_line = np.array([])

print 'input values, the max gradient is', gradients-1
# loop to ask user for input values
for i in range(vals):
    print '\nvalue',i+1,'?'
    val = int(input())
    test_line = np.append(test_line, val)
    
test_line = test_line.astype(np.int32)
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
	test_line = np.subtract(test_line,f_value)

    input_array = np.array([])
    # for each input value, write to the input array
    # with a 1 in the position
    for a in range(vals):
        test_array = np.zeros([1,length])
        test_array[0,test_line[a]] = 1
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
    print 'Probaility of Gradient', np.argmax(probs), '=', probs[0,learnt_ans-l_val]
else:
    print 'Wrong'
    
    # print learnt answer and it probability
    print '\nLearnt answer =', learnt_ans
    print 'Probaility of Gradient', np.argmax(probs), '=', probs[0,learnt_ans-l_val]
    
    # print true answer and its probabilities
    print '\nNumerical answer =', answer
    print 'Probability of Gradient', test_line[1]-test_line[0], '=', probs[0,answer-l_val]
    

# print the probabilities
print '\n\n', probs
