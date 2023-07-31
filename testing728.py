# based on mnist example from jax authors:
##################################################

# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/Neural_Network_and_Data_Loading.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/notebooks/Neural_Network_and_Data_Loading.ipynb)
# 
# **Copyright 2018 The JAX Authors.**
# 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
from jax.scipy.special import logsumexp
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
import optax
import time
import flax
from flax import nn
from jax import random as rnd
from util import *

def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

#layer_sizes = [784, 512, 512, 10]
layer_sizes = [784, 128, 128, 10]
step_size = 0.1
num_epochs = 8
batch_size = 8
n_targets = 10
params = init_network_params(layer_sizes, random.PRNGKey(0))






# ## Auto-batching predictions
# 
# Let us first define our prediction function. Note that we're defining this for a _single_ image example. We're going to use JAX's `vmap` function to automatically handle mini-batches, with no performance penalty.


def relu(x):
  return jnp.maximum(0, x)

def predict(params, image):
  # per-example predictions
  activations = image
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)

  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits - logsumexp(logits)





#class Model(flax.nn.Module):
#  def apply(self, x):
#    x = jnp.reshape(x,(28,28))
#    x = nn.Conv(x, features=32, step_size=2, kernel_size=(3, 3))
#    x = flax.nn.relu(x)
#    x = nn.Conv(x, features=32, step_size=2, kernel_size=(3, 3))
#    x = flax.nn.relu(x)
#    x = jnp.reshape(x,x.shape[:-3]+(-1,))
#    x = flax.nn.Dense(x, features=10)
#    x = flax.nn.log_softmax(x)
#    return x
#
#model=Model()
#params=model.init(rnd.PRNGKey(0), jnp.ones((1,784)))




random_flattened_images = random.normal(random.PRNGKey(1), (10, 28 * 28))

# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))

# `batched_predict` has the same call signature as `predict`
batched_preds = batched_predict(params, random_flattened_images)
print(batched_preds.shape)


def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
  preds = batched_predict(params, images)
  return -jnp.mean(preds * targets)


def get_update_fn(gradfn=None):
  if gradfn is None:
    def gradfn(*args):
      v,g=jax.value_and_grad(loss)(*args)
      return g,v

  def update(params, x, y):
    grads,aux = gradfn(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)], aux
  
  return jax.jit(update)



### new method ###

jac=jax.jacrev(batched_predict)

def newgradfn(params,images,targets):

  O=jac(params,images)

  # make T'
  O_=flattenjac(O,batchdims=2)
  O_=jnp.reshape(O_,(-1,O_.shape[-1]))
  T=jnp.inner(O_,O_)

  l,v=jnp.linalg.eigh(T)
  valid=l>jnp.quantile(l,.6)
  inv=(1/l)*valid
  invT=v*inv[None,:] @ v.T

  e=batched_predict(params,images)-targets
  e=jnp.ravel(e)

  xwise_grads=invT @ e

  def contract(D):
    D=jnp.reshape(D,(-1,)+D.shape[2:])
    D=jnp.moveaxis(D,0,-1)
    return D @ xwise_grads

  return jax.tree_map(contract,O), jnp.linalg.norm(e)

def flattenjac(T,batchdims):
  flatten_array=lambda A:jnp.reshape(A,A.shape[:batchdims]+(-1,))
  T=jax.tree_map(flatten_array,T)
  T,_=jax.tree_flatten(T)
  T=jnp.concatenate(T,axis=-1)
  return T

### end new method ###


# ## Data Loading with PyTorch


def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))


# Define our dataset, using torch datasets
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)

# Get the full train dataset (for checking accuracy while training)
train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)


# ## Training Loop

import sys
import matplotlib.pyplot as plt

#import kfac_jax
#kfac_jax.Optimizer

if 'old' in sys.argv:
  update=get_update_fn()
  mode='old'
else:
  update=get_update_fn(newgradfn)
  mode='new'

losses=[]
accuracies=[]

for epoch in range(num_epochs):
  start_time = time.time()
  for i, (x, y) in enumerate(training_generator):
    y = one_hot(y, n_targets)
    params, aux = update(params, x, y)
    losses.append(aux)
    accuracies.append(accuracy(params, x, y))

    print(losses[-1])
    if i%100==0:
      fig,axs=plt.subplots(2)
      axs[0].plot(losses)
      axs[1].plot(accuracies)
      plt.xscale('log')
      plt.title(mode)
      plt.savefig('loss.png')
      plt.close()

  epoch_time = time.time() - start_time

  train_acc = accuracy(params, train_images, train_labels)
  test_acc = accuracy(params, test_images, test_labels)
  print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))


# We've now used the whole of the JAX API: `grad` for derivatives, `jit` for speedups and `vmap` for auto-vectorization.
# We used NumPy to specify all of our computation, and borrowed the great data loaders from PyTorch, and ran the whole thing on the GPU.
