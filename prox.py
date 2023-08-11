import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
from jax.scipy.special import logsumexp
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST, CIFAR10
import optax
import time
import flax
import flax.linen as nn
from jax import random as rnd
from util import *
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
import sys
from config import *
import time
import datetime
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--a', type=float, default=0.999)
args=parser.parse_args()
mode=args.mode
datasetname=args.dataset
a=args.a


ID=datetime.datetime.now().strftime('%m%d%H%M')
print(ID)

rate = 0.1
num_epochs = 25
n_targets = 10


class Model(nn.Module):
  @nn.compact
  def __call__(self, x, mode='log'):

    if datasetname=='mnist':
      x = jnp.reshape(x,x.shape[:-1]+(28,28,1))
    if datasetname=='cifar10':
      x = jnp.reshape(x,x.shape[:-1]+(32,32,3))

    x = nn.Conv(features=8, strides=2, kernel_size=(3, 3))(x)
    x = nn.LayerNorm()(x)
    x = nn.relu(x)
    x = nn.Conv(features=16, strides=2, kernel_size=(3, 3))(x)
    x = nn.LayerNorm()(x)
    x = nn.relu(x)
    x = jnp.reshape(x,x.shape[:-3]+(-1,))
    x = nn.Dense(features=512)(x)
    x = nn.LayerNorm()(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    
    if mode=='p':
      return nn.softmax(x)
    if mode=='log':
      return nn.log_softmax(x)

model=Model()

batched_predict = model.apply
if datasetname=='mnist':
  dummy_imgs = random.normal(random.PRNGKey(1), (batch_size, 28 * 28))
if datasetname=='cifar10':
  dummy_imgs = random.normal(random.PRNGKey(1), (batch_size, 32*32*3))


dummy_labels = random.normal(random.PRNGKey(1), (batch_size, n_targets))
params=model.init(rnd.PRNGKey(0), dummy_imgs)

def batched_predict_restricted(params,images,targets):
  prediction=batched_predict(params,images,mode='p')
  return jnp.sum(prediction*targets,axis=-1)

jac_restricted=jax.jacrev(batched_predict_restricted)

def one_hot(x, k, dtype=jnp.float32):
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)



#########################
### method with Gil's momentum scheme ###

@jax.jit
def newgradmomentum(params,images,targets,prev_grad):

  O=jac_restricted(params,images,targets)

  def flattenjac(O):
    O=jax.tree_map(lambda A:jnp.reshape(A,(A.shape[0],-1)),O)
    O,_=jax.tree_flatten(O)
    O=jnp.concatenate(O,axis=-1)
    return O

  Ohat=O_=flattenjac(O)
  T=jnp.inner(O_,O_)

  l,v=jnp.linalg.eigh(T)
  valid=l>jnp.quantile(l,.5)
  inv=(1/l)*valid
  invT=v*inv[None,:] @ v.T

  #logits=jnp.sum(batched_predict(params,images)*targets,axis=-1)
  #p=jnp.exp(logits)
  p=jnp.sum(batched_predict(params,images,mode='p')*targets,axis=-1)
  logits=jnp.log(p)
  e=-(1-p)


# Gil's momentum scheme

  prev_grad,_=tree_flatten(prev_grad)
  prev_grad=jnp.concatenate([jnp.ravel(A) for A in prev_grad])

  Ohat_prev_grad = Ohat @ prev_grad
  OhatT_Tinv = Ohat.T @ invT
  
  #

  min_sr_solution = Ohat.T @ (invT @ e)
  prev_grad_subspace = OhatT_Tinv @ Ohat_prev_grad
  prev_grad_complement = prev_grad - prev_grad_subspace
#
  prev_grad_parallel = (
      min_sr_solution
      * (min_sr_solution @ prev_grad_subspace)
      / (min_sr_solution @ min_sr_solution)
  )

  out=(
    0.1*min_sr_solution
    +.9*prev_grad_parallel
    +.99*prev_grad_complement
  )

  treelist,treeshape=tree_flatten(O)
  sizes=[A.size for A in treelist]
  shapes=[A.shape for A in treelist]

  out_=[]
  start=0
  for S,sh in zip(sizes,shapes):
    s=S//batch_size
    block=out[start:start+s]
    out_.append(jnp.reshape(block,sh[1:]))
    start=start+s
  
  return tree_unflatten(treeshape,out_),\
  dict(loss=-jnp.mean(logits)/n_targets)

###########################################################################

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


###########################################################################
# Data Loading with PyTorch


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

import os

if datasetname=='mnist':
  # Define our dataset, using torch datasets
  dataset = MNIST(os.path.join(datasetpath,'mnist/'), download=True, transform=FlattenAndCast())
  training_generator = NumpyLoader(dataset, batch_size=batch_size, num_workers=0)

  # Get the full train dataset (for checking accuracy while training)
  #train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
  #train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

  ## Get full test dataset
  #mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
  #test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
  #test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)


if datasetname=='cifar10':

  # Define our dataset, using torch datasets
  dataset = CIFAR10(os.path.join(datasetpath,'cifar10/'), download=True, transform=FlattenAndCast())
  training_generator = NumpyLoader(dataset, batch_size=batch_size, num_workers=0)

  ## Get the full train dataset (for checking accuracy while training)
  #train_images = np.array(dataset.train_data).reshape(len(dataset.train_data), -1)
  #train_labels = one_hot(np.array(dataset.train_labels), n_targets)

  # Get full test dataset
  #dataset_test = CIFAR10('/tmp/cifar10/', download=True, train=False)
  #test_images = jnp.array(dataset_test.test_data.numpy().reshape(len(dataset_test.test_data), -1), dtype=jnp.float32)
  #test_labels = one_hot(np.array(dataset_test.test_labels), n_targets)


# End data loading
###########################################################################

###########################################################################
# Training Loop


import matplotlib.pyplot as plt
import pickle
import sys

moreinfo=dict()

if mode=='new':
  @jax.jit
  def update(params,grads,rate):
    return tree_map(lambda p,g:p-rate*g,params,grads)
  
if mode=='newadapt':
  opt=optax.adam(learning_rate=00.1)
  optstate=opt.init(params)


if mode=='sgd':
  def loss(params, xy):
    x,y=xy
    logits = batched_predict(params, x)
    return -jnp.mean(logits * y)

  value_and_grad_func=jax.jit(jax.value_and_grad(loss))
  opt=optax.sgd(learning_rate=0.1)
  optstate=opt.init(params)
  
if mode=='kfac':
  import kfac_jax
  def loss(params, xy):
    x,y=xy
    logits = batched_predict(params, x)
    kfac_jax.register_softmax_cross_entropy_loss(logits, y)
    return -jnp.mean(logits * y)

  kfac_opt=kfac_jax.Optimizer(
    value_and_grad_func=jax.value_and_grad(loss),
    l2_reg=.001,
    value_func_has_aux=False,
    value_func_has_state=False,
    value_func_has_rng=False,
    use_adaptive_learning_rate=True,
    use_adaptive_momentum=True,
    use_adaptive_damping=True,
    initial_damping=1.0,
    multi_device=False,
  )
  optstate=kfac_opt.init(params,rnd.PRNGKey(0),(dummy_imgs,dummy_labels))
  key=rnd.PRNGKey(0)


losses=[]
accuracies=[]
prevgrad=tree_map(lambda x:0*x,params)
j=0

for epoch in range(num_epochs):
  start_time = time.time()
  for i, (x, y) in enumerate(training_generator):
    j+=1
    if y.shape!=(batch_size,):
      continue

    y = one_hot(y, n_targets)

    if mode=='new':
      grads, aux = newgradmomentum(params, x, y ,prevgrad)
      prevgrad=grads

      a=0.9999
      rate=0.1*a**j
      moreinfo['a']=a
      params = update(params, grads, rate)

    if mode=='newadapt':
      grads, aux = newgradmomentum(params, x, y ,prevgrad)
      prevgrad=grads
      updates,optstate=opt.update(grads,optstate)
      params=optax.apply_updates(params,updates)

    if mode=='sgd':
      loss_,grads=value_and_grad_func(params,(x,y))
      aux=dict(loss=loss_)
      updates,optstate=opt.update(grads,optstate)
      params=optax.apply_updates(params,updates)

    if mode=='kfac':
      key=rnd.split(key)[0]
      params,optstate,aux=kfac_opt.step(params,optstate,key,batch=(x,y),global_step_int=j)

    losses.append(float(aux['loss']))
    accuracies.append(float(accuracy(params, x, y)))

    print('{}|{}|{}'.format(epoch,i,losses[-1]))
    if i%10==0:

      with open('outputs/{}_{}_{}.pkl'.format(datasetname,mode,ID),'wb') as f:
        pickle.dump(dict(loss=losses,accuracy=accuracies)|moreinfo,f)
