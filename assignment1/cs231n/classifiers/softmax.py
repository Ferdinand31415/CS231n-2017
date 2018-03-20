import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, C = X.shape[0], W.shape[1]
  scores = np.dot(X, W)
  for n in np.arange(N):
    correct_score = scores[n, y[n]]
    norm = np.sum(np.exp(scores[n]))
    for j in range(C):
        #print(W.shape, dW[j].shape, X[n].shape, dW.shape)
        dW[:, j] += X[n] * np.exp(scores[n, j]) / norm
        
    dW[:, y[n]] -= X[n]
    loss -= correct_score - np.log(norm)
  #print('X', X.shape, 'W', W.shape, 'scores', scores.shape)
  
  loss /= N
  loss += reg * np.sum(W * W)
  dW /= N
  dW += 2 * reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  C = W.shape[1]
  features = W.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = np.dot(X, W)
  exp = np.exp(f)
  norm = np.sum(exp, axis=1, keepdims=True)
  loss = -1. / N * (np.sum(f[np.arange(N), y]) - np.sum(np.log(norm)))

  softmax = exp/norm
  softmax[np.arange(N),y] -= 1 #we correct for the substracted x, coming from derivative


  dW += 1. / N * np.dot(X.T, softmax)

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

