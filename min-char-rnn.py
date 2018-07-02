"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""

## add comments by weixsong
## reference page [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## this is a 3 layers neuron network.
## input layer: one hot vector, dim: vocab * 1
## hidden layer: LSTM, hidden vector: hidden_size * 1
## output layer: Softmax, vocab * 1, the probabilities distribution of each character

import numpy as np
import pdb

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file

# use set() to count the vacab size
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))

pdb.set_trace()
# dictionary to convert char to idx, idx to char
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
## RNN/LSTM
## this is not LSTM, is the simple basic RNN
## # update the hidden state
## self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
## # compute the output vector
## y = np.dot(self.W_hy, self.h)
Wx = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden,shape:(100,65)
Wh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden,shape:(100,100)
Wy = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output,shape:(65,100)
bh = np.zeros((hidden_size, 1)) # hidden bias,shape:(100,1)
by = np.zeros((vocab_size, 1)) # output bias,shape:(65,1)


## compute loss, derivative
## cross-entropy loss is used
## actually, here the author use cross-entropy as error,
## but in the backpropagation the author use sum of squared error (Quadratic cost) to do back propagation.
## be careful about this trick. 
## this is because the output layer is a linear layer.
## TRICK: Using the quadratic cost when we have linear neurons in the output layer, z[i] = a[i]
def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.shape:25
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  ## record each hidden state of
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass for each training data point
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size, 1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    
    ## hidden state, using previous hidden state hs[t-1]
    hs[t] = np.tanh(np.dot(Wx, xs[t]) + np.dot(Wh, hs[t-1]) + bh)
    ## unnormalized log probabilities for next chars
    ys[t] = np.dot(Wy, hs[t]) + by
    ## probabilities for next chars, softmax
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
    ## softmax (cross-entropy loss)
    loss += -np.log(ps[t][targets[t], 0])

  # backward pass: compute gradients going backwards
  dWx, dWh, dWy = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wy)#shape:100x65,100x100,65x100
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])#shape:(100,1)
  for t in reversed(range(len(inputs))):
    ## compute derivative of error w.r.t the output probabilites
    ## dE/dy[j] = y[j] - t[j]
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y,https://www.jianshu.com/p/c02a1fbffad6
    
    ## output layer doesnot use activation function, so no need to compute the derivative of error with regard to the net input
    ## of output layer. 
    ## then, we could directly compute the derivative of error with regard to the weight between hidden layer and output layer.
    ## dE/dy[j]*dy[j]/dWy[j,k] = dE/dy[j] * h[k]
    dWy += np.dot(dy, hs[t].T)
    dby += dy
    
    ## backprop into h
    ## derivative of error with regard to the output of hidden layer
    ## derivative of H, come from output layer y and also come from H(t+1), the next time H
    dh = np.dot(Wy.T, dy) + dhnext
    ## backprop through tanh nonlinearity
    ## derivative of error with regard to the input of hidden layer
    ## dtanh(x)/dx = 1 - tanh(x) * tanh(x)
    dhraw = (1 - hs[t] * hs[t]) * dh
    dbh += dhraw
    
    ## derivative of the error with regard to the weight between input layer and hidden layer
    dWx += np.dot(dhraw, xs[t].T)
    dWh += np.dot(dhraw, hs[t-1].T)
    ## derivative of the error with regard to H(t+1)
    ## or derivative of the error of H(t-1) with regard to H(t)
    dhnext = np.dot(Wh.T, dhraw)

  for dparam in [dWx, dWh, dWy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

  return loss, dWx, dWh, dWy, dbh, dby, hs[len(inputs)-1]

## given a hidden RNN state, and a input char id, predict the coming n chars
def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """

  ## a one-hot vector
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1

  ixes = []
  for t in range(n):
    ## self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
    h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h) + bh)
    ## y = np.dot(self.W_hy, self.h)
    y = np.dot(Wy, h) + by
    ## softmax
    p = np.exp(y) / np.sum(np.exp(y))#shape:(65,1)
    ## sample according to probability distribution
    ix = np.random.choice(list(range(vocab_size)), p=p.ravel())

    ## update input x
    ## use the new sampled result as last input, then predict next char again.
    x = np.zeros((vocab_size, 1))
    x[ix] = 1

    ixes.append(ix)

  return ixes


## iterator counter
n = 0
## data pointer
p = 0

mWx, mWh, mWy = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wy)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

## main loop
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p + seq_length + 1 >= len(data) or n == 0:
    # reset RNN memory
    ## hprev is the hiddden state of RNN
    hprev = np.zeros((hidden_size, 1))#shape:(100,1)
    # go from start of data
    p = 0

  inputs = [char_to_ix[ch] for ch in data[p : p + seq_length]]
  targets = [char_to_ix[ch] for ch in data[p + 1 : p + seq_length + 1]]

  # sample from the model now and then
  if n % 100 == 0:
    pdb.set_trace()
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('---- sample -----')
    print('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWx, dWh, dWy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  ## author using Adagrad(a kind of gradient descent)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0:
    print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  ## parameter update for Adagrad is different from gradient descent parameter update
  ## need to learn what is Adagrad exactly is.
  ## seems using weight matrix, derivative of weight matrix and a memory matrix, update memory matrix each iteration
  ## memory is the accumulation of each squared derivatives in each iteration.
  ## mem += dparam * dparam
  for param, dparam, mem in zip([Wx, Wh, Wy, bh, by],
                                [dWx, dWh, dWy, dbh, dby],
                                [mWx, mWh, mWy, mbh, mby]):
    mem += dparam * dparam
    ## learning_rate is adjusted by mem, if mem is getting bigger, then learning_rate will be small
    ## gradient descent of Adagrad
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
