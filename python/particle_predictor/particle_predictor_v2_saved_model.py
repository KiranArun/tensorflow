# import tensorflow and numpy
import tensorflow as tf
import numpy as np


# setup the parameters
# number of different M's
gradients = 10
# number of input values
vals = 3
iterations = 5000
learning_rate = 0.5
model_path = "/tmp/saved_models/model.ckpt"

# defining function to make training data
def training_data():
    
    n = 0
    rows = 1
    # array which we are using as our x values 
    # in equation of linear line, y = Mx
    # it includes 1 extra value as this will be used as our labal
    x = np.arange(vals+1).astype(np.int32)
    # empty array to write our training data to
    y = np.array([])
    
    # loop so it cycles through every gradient
    for i in range(gradients):
        
        y = np.append(y, x*n).reshape(rows,vals+1)
            
        # increase number of rows to reshape it
        rows += 1
        
        # increase gradient by 1
        n+=1
        
    # return the training data
    # and number of lines to learn
    return(y,np.size(y,0))

# print the training data
print(training_data())


# data = training data
# training lines = number of different lines
data,training_lines = training_data()

# the length is for when we convert the numbers into a binary array
# the array will be all zeros except one, which will be 1
# this will be the particle in this pont in time
# each one is like a frame in a video
# this value is the size of the largest M value multiplied by largest x values
length = (gradients-1)*vals-1

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
            in_data[i][a][data[i][a]] = 1
            
        # set the label value to a 1
        lab[i][0][data[i][vals-(vals-1)]] = 1
        
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

saver = tf.train.Saver()

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


# I am using a GPU
# this line limits memory usage of the GPU to 0.4 when session is created
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

# create interactive session using the GPU line for above
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options)) 

# initialize variables
init = tf.initialize_all_variables()
sess.run(init)

# now we run the learning loop
for _ in range(iterations):

    # set training data and labels
    x_data, y_data = set_data()
    # only use one lines data
    # chooses line data by evenly spreading data over iterations
    x_data = x_data[_//(iterations/(training_lines-0.))]
    y_data = y_data[_//(iterations/(training_lines-0.))]
    
    # run the training optimizer
    sess.run(train_step, feed_dict={x: x_data, y_: y_data})

    # print steps and error 20 times in total
    if _ % (iterations/20) == 0:
        print 'step', _, 'out of', iterations
        print 'error =', sess.run(cross_entropy, feed_dict={x: x_data, y_: y_data})




###################################################################################
###################################################################################



save_path = saver.save(sess, model_path)
print("Model saved in file: %s" % save_path)
