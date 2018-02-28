
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 3
# ------------
# 
# Previously in `2_fullyconnected.ipynb`, you trained a logistic regression and a neural network model.
# 
# The goal of this assignment is to explore regularization techniques.

# In[1]:


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle


# First reload the data we generated in `1_notmnist.ipynb`.

# In[2]:


pickle_file = '../Assignment 1/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.

# In[3]:


image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# In[4]:


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# ---
# Problem 1
# ---------
# 
# Introduce and tune L2 regularization for both logistic and neural network models. Remember that L2 amounts to adding a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor `t` using `nn.l2_loss(t)`. The right amount of regularization should improve your validation / test accuracy.
# 
# ---

# In[5]:


batch_size = 128
hidden_units = 1024

graph = tf.Graph()

def model(X, w_h1, b1, w_h2, b2):
  h = tf.nn.relu(tf.matmul(tf_train_dataset, w_h1) + b1)
  return tf.matmul(h, w_h2) + b2

with graph.as_default():
  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)  
  
  # Variables.
  w_h1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_units]))
  b1 = tf.Variable(tf.zeros([hidden_units]))
  w_h2 = tf.Variable(tf.truncated_normal([hidden_units, num_labels]))
  b2 = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  net = model(tf_train_dataset, w_h1, b1, w_h2, b2)

  # Loss Function
  l2_loss = 0.0001 * (tf.nn.l2_loss(w_h1) + tf.nn.l2_loss(w_h2))
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=net)) + l2_loss
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(net)
  valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, w_h1) + b1), w_h2) + b2)
  test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, w_h1) + b1), w_h2) + b2)


# In[6]:


num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# ---
# Problem 2
# ---------
# Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?
# 
# ---

# In[23]:


batch_size = 128
hidden_units = 1024

graph = tf.Graph()

def model(X, w_h1, b1, w_h2, b2):
  h = tf.nn.relu(tf.matmul(tf_train_dataset, w_h1) + b1)
  return tf.matmul(h, w_h2) + b2

with graph.as_default():
  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)  
  
  # Variables.
  w_h1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_units]))
  b1 = tf.Variable(tf.zeros([hidden_units]))
  w_h2 = tf.Variable(tf.truncated_normal([hidden_units, num_labels]))
  b2 = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  net = model(tf_train_dataset, w_h1, b1, w_h2, b2)

  # Loss Function
  l2_loss = 0.0001 * (tf.nn.l2_loss(w_h1) + tf.nn.l2_loss(w_h2))
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=net)) + l2_loss
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  train_prediction = tf.nn.softmax(net)
  valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, w_h1) + b1), w_h2) + b2)
  test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, w_h1) + b1), w_h2) + b2)


# In[27]:


num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (200 - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# ---
# Problem 3
# ---------
# Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides `nn.dropout()` for that, but you have to make sure it's only inserted during training.
# 
# What happens to our extreme overfitting case?
# 
# ---

# In[37]:


batch_size = 128
hidden_units = 1024

graph = tf.Graph()

def model(X, w_h1, b1, w_h2, b2, p_keep_input, p_keep_hidden):
  X = tf.nn.dropout(X, p_keep_input)
  h = tf.nn.relu(tf.matmul(tf_train_dataset, w_h1) + b1)
  #h = tf.nn.dropout(h, p_keep_hidden)
  return tf.matmul(h, w_h2) + b2

with graph.as_default():
  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)  
  
  # Variables.
  w_h1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_units]))
  b1 = tf.Variable(tf.zeros([hidden_units]))
  w_h2 = tf.Variable(tf.truncated_normal([hidden_units, num_labels]))
  b2 = tf.Variable(tf.zeros([num_labels]))
  p_keep_input = tf.placeholder("float")
  p_keep_hidden = tf.placeholder("float")
  # Training computation.
  net = model(tf_train_dataset, w_h1, b1, w_h2, b2, p_keep_input, p_keep_hidden)

  # Loss Function
  l2_loss = 0.0001 * (tf.nn.l2_loss(w_h1) + tf.nn.l2_loss(w_h2))
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=net)) + l2_loss
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(net)
  valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, w_h1) + b1), w_h2) + b2)
  test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, w_h1) + b1), w_h2) + b2)


# In[38]:


num_steps = 5001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, p_keep_input: 0.5, p_keep_hidden: 0.5}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# ---
# Problem 4
# ---------
# 
# Try to get the best performance you can using a multi-layer model! The best reported test accuracy using a deep network is [97.1%](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html?showComment=1391023266211#c8758720086795711595).
# 
# One avenue you can explore is to add multiple layers.
# 
# Another one is to use learning rate decay:
# 
#     global_step = tf.Variable(0)  # count the number of steps taken.
#     learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#  
#  ---

# In[13]:


batch_size = 256
hidden_units = 512

graph = tf.Graph()

def model(X, w_h1, b1, w_h2, b2, w_h3, b3, w_o, b_o, p_keep_input, p_keep_hidden):
  X = tf.nn.dropout(X, p_keep_input)
  h1 = tf.nn.relu(tf.matmul(X, w_h1) + b1)
  h1 = tf.nn.dropout(h1, p_keep_hidden)
  h2 = tf.nn.relu(tf.matmul(h1, w_h2) + b2)
  h2 = tf.nn.dropout(h2, p_keep_hidden)
  h3 = tf.nn.relu(tf.matmul(h2, w_h3) + b3)
  h3 = tf.nn.dropout(h3, p_keep_hidden)
  out = tf.nn.softmax(tf.matmul(h3, w_o) + b_o)
  return out

def validate_model(X, w_h1, b1, w_h2, b2, w_h3, b3, w_o, b_o):
    h1 = tf.nn.relu(tf.matmul(X, w_h1)+ b1)
    h2 = tf.nn.relu(tf.matmul(h1, w_h2)+ b2)
    h3 = tf.nn.relu(tf.matmul(h2, w_h3)+ b3)
    out = (tf.matmul(h3, w_o)+ b_o)
    return tf.nn.softmax(out)
                         
with graph.as_default():
  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)  
  
  # Variables.
  w_h1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_units]))
  b1 = tf.Variable(tf.zeros([hidden_units]))
  w_h2 = tf.Variable(tf.truncated_normal([hidden_units, hidden_units]))
  b2 = tf.Variable(tf.zeros([hidden_units]))
  w_h3 = tf.Variable(tf.truncated_normal([hidden_units, hidden_units]))
  b3 = tf.Variable(tf.zeros([hidden_units]))
  w_o = tf.Variable(tf.truncated_normal([hidden_units, num_labels]))
  b_o = tf.Variable(tf.zeros([num_labels]))
  p_keep_input = tf.placeholder("float")
  p_keep_hidden = tf.placeholder("float")
  
  # Training computation.
  net = model(tf_train_dataset, w_h1, b1, w_h2, b2, w_h3, b3, w_o, b_o, p_keep_input, p_keep_hidden)

  # Loss Function
  l2_loss = 0.001 * (tf.nn.l2_loss(w_h1) + tf.nn.l2_loss(w_h2) + tf.nn.l2_loss(w_h3) + tf.nn.l2_loss(w_o))
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=net)) + l2_loss
  
  # Optimizer.
  #optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.5, global_step, 10000, 0.95)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(net)
  valid_prediction = validate_model(tf_valid_dataset, w_h1, b1, w_h2, b2, w_h3, b3, w_o, b_o)
  test_prediction = validate_model(tf_test_dataset, w_h1, b1, w_h2, b2, w_h3, b3, w_o, b_o)


# In[14]:


num_steps = 20001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, p_keep_input: 0.5, p_keep_hidden: 0.5}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 1000 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

