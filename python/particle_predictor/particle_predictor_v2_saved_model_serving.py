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


# setup the parameters
# number of different M's
gradients = 10
# number of input values
vals = 3
iterations = 2000
learning_rate = 0.5

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
    
    y= y.astype(np.int32)
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

# we will feed the labels in here
y_ = tf.placeholder(tf.float32, [None, gradients])

# configure the loss function, using cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# define the optimizer
# AdagradOptimizer works much better than GradientDescentOptimizer
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)

serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
prediction_classes = y
values = y
###################################################################################
###################################################################################


# I am using a GPU
# this line limits memory usage of the GPU to 0.4 when session is created
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

# create interactive session using the GPU line for above
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options)) 

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# now we run the learning loop
for _ in range(iterations):

    # set training data and labels
    x_data, y_data = set_data()
    # only use one lines data
    # chooses line data by evenly spreading data over iterations
    x_data = x_data[np.round(_//(iterations/(training_lines-0.)), 0).astype(np.int32)]
    y_data = y_data[np.round(_//(iterations/(training_lines-0.)), 0).astype(np.int32)]
    
    # run the training optimizer
    sess.run(train_step, feed_dict={x: x_data, y_: y_data})

    # print steps and error 20 times in total
    if _ % (iterations/20) == 0:
        print 'step', _, 'out of', iterations
        print 'error =', sess.run(cross_entropy, feed_dict={x: x_data, y_: y_data})




###################################################################################
###################################################################################


export_path_base = "/my-files/tmp/saved_models/"

export_path = os.path.join(compat.as_bytes(export_path_base),compat.as_bytes(str(1)))
      
print 'Exporting trained model to', export_path
builder = saved_model_builder.SavedModelBuilder(export_path)

# Build the signature_def_map.
classification_inputs = utils.build_tensor_info(serialized_tf_example)
classification_outputs_classes = utils.build_tensor_info(prediction_classes)
classification_outputs_scores = utils.build_tensor_info(values)

classification_signature = signature_def_utils.build_signature_def(
      inputs={signature_constants.CLASSIFY_INPUTS: classification_inputs},
      outputs={
          signature_constants.CLASSIFY_OUTPUT_CLASSES:
              classification_outputs_classes,
          signature_constants.CLASSIFY_OUTPUT_SCORES:
              classification_outputs_scores
      },
      method_name=signature_constants.CLASSIFY_METHOD_NAME)

tensor_info_x = utils.build_tensor_info(x)
tensor_info_y = utils.build_tensor_info(y)

prediction_signature = signature_def_utils.build_signature_def(
      inputs={'frames': tensor_info_x},
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
