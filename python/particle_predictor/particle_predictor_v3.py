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
iterations = 4000
# limit gpu memory usage
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
learning_rate = 1e-5


#  we create input data as numbers with extra for label
def training_data():
    
    M = 0
    rows = 1
    # X and Y values of equation y = Mx+b
    X = np.arange(vals+1)
    Y = np.array([])
    
    # append new line for each gradient and reshape
    for i in range(gradients):
        
        Y = np.append(Y, X*M).reshape(rows,vals+1)
            
        rows += 1
        M+=1

    return(Y.astype(np.int32),np.size(Y,0))

#print(training_data())

# y arrays and number of them form before
training_line_data,training_lines = training_data()

# set lengths to set when formatting input data
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
        first_value = training_line_data[0]
        training_line_data = np.subtract(training_line_data,first_value)
        # create input data
        input_data = np.zeros([vals,length])
        
    # loop over nummber of lines
    for i in range(number_lines):
        # loop over number of values to plot
        for a in range(number_points):
            # set the specified input data values to 1
            if mode == 'train':
                input_data[i,a,training_line_data[i,a]] = 1
            else:
                input_data[a,training_line_data[a]] = 1
            
        # set the specified label values to 1 if in train mode
        if mode == 'train':
            labels[i,0,training_line_data[i,1]] = 1
    
    # reshape and return data
    if mode == 'train':
        input_data = input_data.reshape(number_lines,1,full_length)
        return(input_data.astype(np.float32),labels)
    else:
        input_data = input_data.reshape(1,full_length)
        return(input_data.astype(np.float32))


#print(format_data(training_line_data, 'train'))

# define functions to create weights/biases
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


x = tf.placeholder(tf.float32, [None, full_length])

y_ = tf.placeholder(tf.float32, [None, gradients])

y_conv, keep_prob = deepnn(x)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
   
sess.run(tf.global_variables_initializer())

for i in range(iterations):

    x_data, y_data = format_data(training_line_data, 'train')
    x_acc_test = x_data.reshape(training_lines,-1)
    y_acc_test = y_data.reshape(training_lines,-1)
    
    x_data = x_data[i%training_lines]
    y_data = y_data[i%training_lines]
    
    if i % (iterations/20) == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:x_acc_test, y_: y_acc_test, keep_prob: 1.0})
        error = sess.run(cross_entropy, feed_dict={x:x_data, y_: y_data, keep_prob: 1.0})
        
        print 'step', i, 'training accuracy', train_accuracy,  '\nerror =', error
        
    train_step.run(feed_dict={x: x_data, y_: y_data, keep_prob: 0.5})
   
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


probs = sess.run(y_conv, feed_dict={x:input_array, keep_prob:1.0})
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
    print 'Probaility of Gradient', np.argmax(probs), '=', probs[0][learnt_ans-last_val]
else:
    print 'Wrong'
    
    # print learnt answer and it probability
    print '\nLearnt answer =', learnt_ans
    print 'Probaility of Gradient', np.argmax(probs), '=', probs[0][learnt_ans-last_val]
    
    # print true answer and its probabilities
    print '\nNumerical answer =', answer
    print 'Probability of Gradient', test_line[1]-test_line[0], '=', probs[0][answer-last_val]
    

# print the probabilities
print '\n\n', probs


