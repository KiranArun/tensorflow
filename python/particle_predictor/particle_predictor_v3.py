# import tensorflow and numpy
import tensorflow as tf
import numpy as np


# Hyperparameters
# number of points for input data
vals = 4
# size of frame (1d in this case)
max_answer = 100

# number of gradients to learn, it goes up to
# the maximum linear line for this size of frame
gradients = max_answer/(vals)+1

iterations = 10000

# limit gpu memory usage
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
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

#print(training_data())

# y arrays and number of them form before
training_line_data,training_lines = training_data()

# set lengths to use when formatting input data
length = max_answer
full_length = length*vals

# function to format data
# variables are line data in number form and mode
def format_data(training_line_data, mode):
            
    # if its in training mode
    if mode == 'train':
        # set number of times to loop
        number_points = vals
        number_lines = training_lines
        # create labels and input data
        labels = np.zeros([training_lines,1,gradients])
        input_data = np.zeros([training_lines,vals,length])
        
    else:
        # set number of times to loop
        number_points = vals
        number_lines = 1
        # set first value to take away from the array to remove b
        first_value = test_line[0]
        training_line_data = np.subtract(training_line_data,first_value)
        # create input data
        input_data = np.zeros([vals,length])
        
    # loop for each different line
    for i in range(number_lines):
        # loop for each value to plot
        for a in range(number_points):
            
            # set the specified input data values to 1
            if mode == 'train':
                input_data[i,a,training_line_data[i,a]] = 1
            else:
                # only do it for one line if not in training mode
                input_data[a,training_line_data[a]] = 1
            
        # set the specified label values to 1 if in train mode
        if mode == 'train':
            labels[i,0,training_line_data[i,1]] = 1
    
    # reshape and return data and labels if in train mode
    if mode == 'train':
        input_data = input_data.reshape(number_lines,1,full_length)
        return(input_data.astype(np.float32),labels)
    else:
        input_data = input_data.reshape(1,full_length)
        return(input_data.astype(np.float32))

# print data if necessary
#print(format_data(training_line_data, 'train'))




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


# x placeholder for input data
x = tf.placeholder(tf.float32, [None, full_length])
# y_ placeholder for labels
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

# run session
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
   
# initialize variables
sess.run(tf.global_variables_initializer())

# training loop
for i in range(iterations):

    # select data for training and for accuracy testing
    x_data, y_data = format_data(training_line_data, 'train')
    
    # this chooses the whole batch (since its very small) and later uses it for accuracy testing
    x_acc_test = x_data.reshape(training_lines,-1)
    y_acc_test = y_data.reshape(training_lines,-1)
    
    # this is the training data and it picks just one line in the whole data set
    x_data = x_data[i%training_lines]
    y_data = y_data[i%training_lines]

    # print accuracy and error 20 times
    if i % (iterations/20) == 0:
        # set accuracy and error by running accuracy from before and error from cross entropy
        train_accuracy = sess.run(accuracy, feed_dict={x:x_acc_test, y_: y_acc_test, keep_prob: 1.0})
        error = sess.run(cross_entropy, feed_dict={x:x_data, y_: y_data, keep_prob: 1.0})
        
        # print accuracy and error
        print 'step', i, 'training accuracy', train_accuracy,  '\nerror =', error
        
    # run the training for  this iteration
    train_step.run(feed_dict={x: x_data, y_: y_data, keep_prob: 0.5})
    
# once training is complete, test teh accuracy for the final time
x_data, y_data = format_data(training_line_data, 'train')

x_acc_test = x_data.reshape(training_lines,-1)
y_acc_test = y_data.reshape(training_lines,-1)

print("test accuracy %g" % sess.run(accuracy, feed_dict={x: x_acc_test, y_: y_acc_test, keep_prob: 1.0}))



# initialize array to record test values
test_line = np.array([])

print 'input values, the max gradient is', gradients-1
# loop to ask user for input values
for i in range(vals):
    print '\nvalue',i+1,'?'
    val = int(input())
    test_line = np.append(test_line, val)
    
# set test line to int32 and setlast value
test_line = test_line.astype(np.int32)
last_val = test_line[-1]

# use format data function to set data to binary
input_array = format_data(test_line, 0)
#print input_array


probs = sess.run(y_conv, feed_dict={x:input_array, keep_prob:1.})
# now we have found the predicted gradient,
# we add the last value to get the next value
learnt_ans = np.argmax(probs)+last_val
# "true" answer, should work with all linear lines
# this numerically works out hte gradient by findingthe difference,
# then also adds the last value
answer = (test_line[1]-test_line[0])+last_val

if answer == learnt_ans:
    print 'Correct'
    
    # print learnt answer and it probability
    print '\nLearnt answer =', learnt_ans
    print 'Probaility of Gradient', np.argmax(probs), '=', probs[0,learnt_ans-last_val]
else:
    print 'Wrong'
    
    # print learnt answer and it probability
    print '\nLearnt answer =', learnt_ans
    print 'Probaility of Gradient', np.argmax(probs), '=', probs[0,learnt_ans-last_val]
    
    # print true answer and its probabilities
    print '\nNumerical answer =', answer
    print 'Probability of Gradient', test_line[1]-test_line[0], '=', probs[0,answer-last_val]
    

# print the probabilities
print '\n\n', probs
