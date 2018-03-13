import numpy as np
from random import shuffle
#from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    #if i==0:
    #  print(scores[:20], np.mean(W))
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      #if i==0: print(j, margin)

      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #print(dW.shape, dW[j].shape, X[i].shape)
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss and gradient.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  N = X.shape[0]
  dW = np.zeros(W.shape) # initialize the gradient as zero
  S = np.dot(X, W) #scores
 
  import time
  #Sy = np.diag(S[:,y]).reshape((y.shape[0], 1)) #about 3.3 times longer
  Sy = S[np.arange(N), y].reshape(-1, 1)

  margin = S - Sy + 1
  #second try
  margin[np.arange(N), y] = 0
  M = np.maximum(0, margin) #Maximums
  loss = 1.0 / N * np.sum(M)
  
  M[M>0] = 1
  M[np.arange(N), y] -= np.sum(M, axis=1) #we subtract the inner derivative resulting from d(Sy) / d(W).
  dW += np.dot(X.T, M)
  '''
  #first try:
  maxi = np.maximum(0, margin)
  loss = 1.0 / N * np.sum(maxi) - 1
  
  maxi[maxi > 0] = 1 #normalize everything to one
  theta = maxi
  dW += np.dot(X.T, theta)
  
  #one hot
  delta = np.zeros((N, 10))
  delta[np.arange(N), y] = 1
  theta_bar = np.sum(theta, axis=1)
  
  x = X.T[...,np.newaxis]
  tb = theta_bar[np.newaxis,:,np.newaxis]
  d = delta[np.newaxis,...]
  SUM = x * tb * d
  dW -= np.sum(SUM, axis=1)
  '''
  dW /= N
  
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W #regularization

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
