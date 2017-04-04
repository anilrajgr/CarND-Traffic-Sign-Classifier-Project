#!/home/anilraj/anaconda3/bin/python

import sys, getopt
import pickle
import os

epochs = 10
batchsize = 128
mu = 0
sigma = 0.01
conv1depth = 6
conv2depth = 16
fc1size = 120
fc2size = 84
conv1keepprob = 0.9
conv2keepprob = 0.9
conv1adepth = 3
conv2adepth = 8
fc1asize = 60
fc2asize = 42
conv1akeepprob = 0.9
conv2akeepprob = 0.9
learningrate = 0.001

outputs2avg = 5

try:
    myopts, args = getopt.getopt(sys.argv[1:],"a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:")
except getopt.GetoptError as e:
    print (str(e))
    sys.exit(2)

for o, a in myopts:
    if o == '-a':
        epochs=int(a)
    elif o == '-b':
        batchsize=int(a)
    elif o == '-c':
        mu=float(a)
    elif o == '-d':
        sigma=float(a)
    elif o == '-e':
        conv1depth=int(a)
    elif o == '-f':
        conv2depth=int(a)
    elif o == '-g':
        fc1size=int(a)
    elif o == '-h':
        fc2size=int(a)
    elif o == '-i':
        conv1keepprob=float(a)
    elif o == '-j':
        conv2keepprob=float(a)
    elif o == '-k':
        conv1adepth=int(a)
    elif o == '-l':
        conv2adepth=int(a)
    elif o == '-m':
        fc1asize=int(a)
    elif o == '-n':
        fc2asize=int(a)
    elif o == '-o':
        conv1akeepprob=float(a)
    elif o == '-p':
        conv2akeepprob=float(a)
    elif o == '-q':
        learningrate=float(a)

# print("epochs: %s, batchsize: %s, mu: %s, sigma: %s, conv1depth: %s, conv2depth: %s, fc1size: %s, fc2size: %s, conv1keepprob: %s, conv2keepprob: %s, learningrate: %s\n" % (epochs, batchsize, mu, sigma, conv1depth, conv2depth, fc1size, fc2size, conv1keepprob, conv2keepprob, learningrate))
# exit()

epochs = 100
batchsize = 64

wfile = open('outdir2/' + str(os.getpid()) + ".txt", "w")
wfile.write("epochs: %s, batchsize: %s, mu: %s, sigma: %s, conv1depth: %s, conv2depth: %s, fc1size: %s, fc2size: %s, conv1keepprob: %s, conv2keepprob: %s, conv1adepth: %s, conv2adepth: %s, fc1asize: %s, fc2asize: %s, conv1akeepprob: %s, conv2akeepprob: %s, learningrate: %s\n" % (epochs, batchsize, mu, sigma, conv1depth, conv2depth, fc1size, fc2size, conv1keepprob, conv2keepprob, conv1adepth, conv2adepth, fc1asize, fc2asize, conv1akeepprob, conv2akeepprob, learningrate))
print("epochs: %s, batchsize: %s, mu: %s, sigma: %s, conv1depth: %s, conv2depth: %s, fc1size: %s, fc2size: %s, conv1keepprob: %s, conv2keepprob: %s, conv1adepth: %s, conv2adepth: %s, fc1asize: %s, fc2asize: %s, conv1akeepprob: %s, conv2akeepprob: %s, learningrate: %s" % (epochs, batchsize, mu, sigma, conv1depth, conv2depth, fc1size, fc2size, conv1keepprob, conv2keepprob, conv1adepth, conv2adepth, fc1asize, fc2asize, conv1akeepprob, conv2akeepprob, learningrate))

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

# Shuffle the data
from sklearn.utils import shuffle
    
X_train, y_train = shuffle(train['features'], train['labels'])
X_valid, y_valid = shuffle(valid['features'], valid['labels'])
X_test, y_test = shuffle(test['features'], test['labels'])


"""
# RGB to YUB
import cv2
for i in range(0, len(X_train)):
  X_train[i] = cv2.cvtColor(X_train[i], cv2.COLOR_BGR2YCrCb)
for i in range(0, len(X_valid)):
  X_valid[i] = cv2.cvtColor(X_valid[i], cv2.COLOR_BGR2YCrCb)
for i in range(0, len(X_test)):
  X_test[i] = cv2.cvtColor(X_test[i], cv2.COLOR_BGR2YCrCb)

# Pick only Y. Possibly Gray scale
X_train = X_train[:,:,:,1]
X_train = X_train.reshape((len(X_train), 32, 32, 1))
X_valid = X_valid[:,:,:,1]
X_valid = X_valid.reshape((len(X_valid), 32, 32, 1))
X_test = X_test[:,:,:,1]
X_test = X_test.reshape((len(X_test), 32, 32, 1))
"""


image_depth = X_train.shape[3]

import numpy as np

from sklearn import preprocessing

# print(X_train)
# X_train_flatten = X_train.reshape((len(X_train), 32*32*image_depth))
# print(X_train_flatten.shape)
# X_train_flatten = preprocessing.scale(X_train_flatten)
# print(np.mean(X_train_flatten, axis=0))
# print(X_train_flatten.shape)
# # print(X_train_flatten)
# # exit()
# X_train = X_train_flatten.reshape((len(X_train), 32, 32, image_depth))
# print(X_train)
# exit()

X_train_flatten = X_train.reshape((len(X_train), 32*32*image_depth))
X_valid_flatten = X_valid.reshape((len(X_valid), 32*32*image_depth))
X_test_flatten = X_test.reshape((len(X_test), 32*32*image_depth))

scaler = preprocessing.StandardScaler().fit(X_train_flatten)

X_train_flatten = scaler.transform(X_train_flatten)
X_train = X_train_flatten.reshape((len(X_train), 32, 32, image_depth))
X_valid_flatten = scaler.transform(X_valid_flatten)
X_valid = X_valid_flatten.reshape((len(X_valid), 32, 32, image_depth))
X_test_flatten = scaler.transform(X_test_flatten)
X_test = X_test_flatten.reshape((len(X_test), 32, 32, image_depth))

# print(X_test)
# print(X_test.shape)
# print(X_test[:][:][:][1])
# print(X_test[:,:,:,1])
# print(X_test[:,:,:,1].shape)
# print(X_test[:,:,:,0].reshape(len(X_test)*32*32, 1))
# print(np.reshape(X_test, (12630, 32, 96)))
# x = X_test[:,:,:,0].reshape(len(X_test)*32*32, 1)
# x = x.astype(np.float32, copy=False)
# print(x)
# x = preprocessing.normalize(x)
# for y in x:
  # print(y)
# # print(X_test[:,:,:,0].reshape(len(X_test)*32*32, 1))

# X_train = preprocessing.normalize(X_train, norm='l2')
# X_valid = preprocessing.normalize(X_train, norm='l2')
# X_test = preprocessing.normalize(X_train, norm='l2')
# 

# TODO: Number of training examples
n_train = len(train['features'])

# TODO: Number of testing examples.
n_test = len(test['features'])

## Image size 32x32
# print(len(train['features'][0][0])) # = 32
# print(len(train['features'][0])) # = 32

# print(train['features'][0].shape)

# TODO: What's the shape of an traffic sign image?
image_shape = train['features'][0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(train['labels']))

# print("Number of training examples =", n_train)
# print("Number of testing examples =", n_test)
# print("Image data shape =", image_shape)
# print("Number of classes =", n_classes)


### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
from random import randint


# X_train, y_train = shuffle(train['features'], train['labels'])

import tensorflow as tf
EPOCHS = epochs
BATCH_SIZE = batchsize

tf.logging.set_verbosity(tf.logging.ERROR)

### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten

def LeNet(x, keep_prob1, keep_prob2, keep_prob1a, keep_prob2a):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    # mu = 0
    # sigma = 0.01
    
    print(x.shape)
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, image_depth, conv1depth), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(conv1depth))
    LeNet.conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    
    # Activation.
    LeNet.conv1 = tf.nn.relu(LeNet.conv1)
    LeNet.conv1 = tf.nn.dropout(LeNet.conv1, keep_prob1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    LeNet.conv1 = tf.nn.max_pool(LeNet.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x60.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, conv1depth, conv2depth), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(conv2depth))
    LeNet.conv2   = tf.nn.conv2d(LeNet.conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    LeNet.conv2 = tf.nn.relu(LeNet.conv2)
    LeNet.conv2 = tf.nn.dropout(LeNet.conv2, keep_prob2)

    # Pooling. Input = 10x10x60. Output = 5x5x60.
    LeNet.conv2 = tf.nn.max_pool(LeNet.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x60. Output = 1500.
    fc0   = flatten(LeNet.conv2)
    
    # Layer 3: Fully Connected. Input = 1500. Output = fc1size.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(5*5*conv2depth, fc1size), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(fc1size))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = fc1size. Output = fc2size.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(fc1size, fc2size), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(fc2size))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = fc2size. Output = n_classes.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(fc2size, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1a_W = tf.Variable(tf.truncated_normal(shape=(5, 5, image_depth, conv1adepth), mean = mu, stddev = sigma))
    conv1a_b = tf.Variable(tf.zeros(conv1adepth))
    LeNet.conv1a   = tf.nn.conv2d(x, conv1a_W, strides=[1, 1, 1, 1], padding='VALID') + conv1a_b
    
    # Activation.
    LeNet.conv1a = tf.nn.relu(LeNet.conv1a)
    LeNet.conv1a = tf.nn.dropout(LeNet.conv1a, keep_prob1a)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    LeNet.conv1a = tf.nn.max_pool(LeNet.conv1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x60.
    conv2a_W = tf.Variable(tf.truncated_normal(shape=(5, 5, conv1adepth, conv2adepth), mean = mu, stddev = sigma))
    conv2a_b = tf.Variable(tf.zeros(conv2adepth))
    LeNet.conv2a   = tf.nn.conv2d(LeNet.conv1a, conv2a_W, strides=[1, 1, 1, 1], padding='VALID') + conv2a_b
    
    # Activation.
    LeNet.conv2a = tf.nn.relu(LeNet.conv2a)
    LeNet.conv2a = tf.nn.dropout(LeNet.conv2a, keep_prob2a)

    # Pooling. Input = 10x10x60. Output = 5x5x60.
    LeNet.conv2a = tf.nn.max_pool(LeNet.conv2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x60. Output = 1500.
    fc0a   = flatten(LeNet.conv2a)
    
    # Layer 3: Fully Connected. Input = 1500. Output = fc1size.
    fc1a_W = tf.Variable(tf.truncated_normal(shape=(5*5*conv2adepth, fc1asize), mean = mu, stddev = sigma))
    fc1a_b = tf.Variable(tf.zeros(fc1asize))
    fc1a   = tf.matmul(fc0a, fc1a_W) + fc1a_b
    
    # Activation.
    fc1a    = tf.nn.relu(fc1a)

    # Layer 4: Fully Connected. Input = fc1size. Output = fc2size.
    fc2a_W  = tf.Variable(tf.truncated_normal(shape=(fc1asize, fc2asize), mean = mu, stddev = sigma))
    fc2a_b  = tf.Variable(tf.zeros(fc2asize))
    fc2a    = tf.matmul(fc1a, fc2a_W) + fc2a_b
    
    # Activation.
    fc2a    = tf.nn.relu(fc2a)

    # Layer 5: Fully Connected. Input = fc2size. Output = n_classes.
    fc3a_W  = tf.Variable(tf.truncated_normal(shape=(fc2asize, n_classes), mean = mu, stddev = sigma))
    fc3a_b  = tf.Variable(tf.zeros(n_classes))


    logits = tf.matmul(fc2, fc3_W) + fc3_b + tf.matmul(fc2a, fc3a_W) + fc3a_b
    
    return logits

def AnilNet(x, keep_prob1, keep_prob2):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    # mu = 0
    # sigma = 0.01
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, image_depth, conv1depth), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(conv1depth))
    LeNet.conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    
    # Activation.
    LeNet.conv1 = tf.nn.relu(LeNet.conv1)
    LeNet.conv1 = tf.nn.dropout(LeNet.conv1, keep_prob1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    LeNet.conv1 = tf.nn.max_pool(LeNet.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x60.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, conv1depth, conv2depth), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(conv2depth))
    LeNet.conv2   = tf.nn.conv2d(LeNet.conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    LeNet.conv2 = tf.nn.relu(LeNet.conv2)
    LeNet.conv2 = tf.nn.dropout(LeNet.conv2, keep_prob2)

    # Pooling. Input = 10x10x60. Output = 5x5x60.
    LeNet.conv2 = tf.nn.max_pool(LeNet.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x60. Output = 1500.
    fc0   = flatten(LeNet.conv2)
    
    # Layer 3: Fully Connected. Input = 1500. Output = fc1size.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(5*5*conv2depth, fc1size), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(fc1size))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = fc1size. Output = fc2size.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(fc1size, fc2size), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(fc2size))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = fc2size. Output = n_classes.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(fc2size, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


### Feel free to use as many code cells as needed.

x = tf.placeholder(tf.float32, (None, 32, 32, image_depth))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
keep_prob1 = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
keep_prob1a = tf.placeholder(tf.float32)
keep_prob2a = tf.placeholder(tf.float32)

rate = learningrate

logits = LeNet(x, keep_prob1, keep_prob2, keep_prob1a, keep_prob2a)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

lastNacc = np.zeros((outputs2avg))

# print(X_train.shape)
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob1: conv1keepprob, keep_prob2: conv2keepprob, keep_prob1a: conv1akeepprob, keep_prob2a: conv2akeepprob})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    # print("Training...")
    # print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob1: 1.0, keep_prob2: 1.0, keep_prob1a: 1.0, keep_prob2a: 1.0})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        lastNacc[1:] = lastNacc[:-1]
        lastNacc[0] = validation_accuracy
        lastNvalidation_accuracy = np.mean(lastNacc)
        # print("EPOCH {} ...".format(i+1))
        wfile.write("{}: Validation Accuracy = {:.3f} ({:.3f})\n".format(i+1, validation_accuracy, lastNvalidation_accuracy))
        print("{}: Validation Accuracy = {:.3f} ({:.3f})".format(i+1, validation_accuracy, lastNvalidation_accuracy))
        if validation_accuracy <= lastNvalidation_accuracy:
          print("Breaking")
          break
        # print()

    test_accuracy = evaluate(X_test, y_test)
    wfile.write("Test Accuracy = {:.3f}\n".format(test_accuracy))
    print("Test Accuracy = {:.3f}".format(test_accuracy))

wfile.close()
