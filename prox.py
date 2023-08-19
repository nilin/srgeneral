import jax.numpy as jnp
from jax import random
import jax
from jax.scipy.special import logsumexp
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST, CIFAR10
import optax
import time
import flax.linen as nn
from jax import random as rnd
from util import *
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from localconfig import *
import time
import datetime
import json
import argparse
import os
import kfac_jax
import dataloading

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--lr', type=float)
args=parser.parse_args()
mode=args.mode
dataset=args.dataset
lr=args.lr

config=dict(
  notes='{} {} lr={}'.format(mode,dataset,lr),
  mode=mode,
  dataset=dataset,
  lr=lr,
  batch_size=batch_size,
  )

ID=datetime.datetime.now().strftime('%m%d-%H%M%S')
print(ID)

logdir=f'logs/{ID}'
os.makedirs(logdir)
json.dump(config,open(f'{logdir}/config.json','w'))


loader,(dummy_imgs,dummy_labels)=dataloading.getloader(dataset)

num_epochs = 25
n_targets = 10

class Model(nn.Module):
  @nn.compact
  def __call__(self, x):

    if dataset=='mnist':
      x = jnp.reshape(x,x.shape[:-1]+(28,28,1))
    if dataset=='cifar10':
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
    
    return nn.log_softmax(x)

model=Model()

batched_predict = model.apply
params=model.init(rnd.PRNGKey(0), dummy_imgs)

def batched_predict_restricted(params,images,targets):
  prediction=batched_predict(params,images)
  return jnp.sum(prediction*targets,axis=-1)

jac_restricted=jax.jacrev(batched_predict_restricted)

def one_hot(x, k, dtype=jnp.float32):
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)


#########################
# ProxSR

@jax.jit
def proxsr(params,images,targets,prev_grad):

  O=jac_restricted(params,images,targets)

  def flattenjac(O):
    O=jax.tree_map(lambda A:jnp.reshape(A,(A.shape[0],-1)),O)
    O,_=jax.tree_flatten(O)
    O=jnp.concatenate(O,axis=-1)
    return O

  Ohat=O_=flattenjac(O)
  T=jnp.inner(O_,O_)

  eps=1e-3
  scale=jnp.trace(T)/T.shape[0]
  invT=jnp.linalg.inv(T+eps*scale*jnp.eye(T.shape[0]))

  targetlogits=jnp.sum(batched_predict(params,images)*targets,axis=-1)
  e=targetlogits

  #ProxSR

  prev_grad,_=tree_flatten(prev_grad)
  prev_grad=jnp.concatenate([jnp.ravel(A) for A in prev_grad])

  Ohat_prev_grad = Ohat @ prev_grad
  OhatT_Tinv = Ohat.T @ invT
  
  min_sr_solution = Ohat.T @ (invT @ e)
  prev_grad_subspace = OhatT_Tinv @ Ohat_prev_grad
  prev_grad_complement = prev_grad - prev_grad_subspace

  out=(
    min_sr_solution
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
  
  return (
    -jnp.mean(targetlogits)/n_targets,
    tree_unflatten(treeshape,out_),
  )

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
## See the License for the specific language governing permissions and
## limitations under the License.
#
#
############################################################################
## Data Loading with PyTorch
#
#def numpy_collate(batch):
#  if isinstance(batch[0], np.ndarray):
#    return np.stack(batch)
#  elif isinstance(batch[0], (tuple,list)):
#    transposed = zip(*batch)
#    return [numpy_collate(samples) for samples in transposed]
#  else:
#    return np.array(batch)
#
#class NumpyLoader(data.DataLoader):
#  def __init__(self, dataset, batch_size=1,
#                shuffle=False, sampler=None,
#                batch_sampler=None, num_workers=0,
#                pin_memory=False, drop_last=False,
#                timeout=0, worker_init_fn=None):
#    super(self.__class__, self).__init__(dataset,
#        batch_size=batch_size,
#        shuffle=shuffle,
#        sampler=sampler,
#        batch_sampler=batch_sampler,
#        num_workers=num_workers,
#        collate_fn=numpy_collate,
#        pin_memory=pin_memory,
#        drop_last=drop_last,
#        timeout=timeout,
#        worker_init_fn=worker_init_fn)
#
#class FlattenAndCast(object):
#  def __call__(self, pic):
#    return np.ravel(np.array(pic, dtype=jnp.float32))
#
#if dataset=='mnist':
#  thedataset = MNIST(os.path.join(datasetpath,'mnist/'), download=True, transform=FlattenAndCast())
#
#if dataset=='cifar10':
#  thedataset = CIFAR10(os.path.join(datasetpath,'cifar10/'), download=True, transform=FlattenAndCast())
#
#training_generator = NumpyLoader(thedataset, batch_size=batch_size, num_workers=0)
#
# End data loading
###########################################################################
# general loss

def lossfn(params, x, y, *args):
  logits = batched_predict(params, x)
  kfac_jax.register_softmax_cross_entropy_loss(logits, y)
  return -jnp.mean(logits * y)

valgradfn=jax.jit(jax.value_and_grad(lossfn))

###########################################################################

zerograd=tree_map(lambda x:0*x,params)
prevgrad=zerograd

if mode=='ProxSR':
  valgradfn=proxsr
  optimizer=optax.sgd(learning_rate=lr)
  optstate=optimizer.init(params)
  
if mode=='sgd':
  optimizer=optax.sgd(learning_rate=lr)
  optstate=optimizer.init(params)

if mode=='adam':
  optimizer=optax.adam(learning_rate=lr)
  optstate=optimizer.init(params)

if mode=='kfac':
  optimizer=kfac_jax.Optimizer(
    value_and_grad_func=jax.value_and_grad(lambda params,xy: lossfn(params,xy[0],xy[1])),
    l2_reg=.001,
    value_func_has_aux=False,
    value_func_has_state=False,
    value_func_has_rng=False,
    #learning_rate_schedule=optax.constant_schedule(lr),
    use_adaptive_learning_rate=True,
    use_adaptive_momentum=True,
    use_adaptive_damping=True,
    initial_damping=1.0,
    multi_device=False,
  )
  optstate=optimizer.init(params,rnd.PRNGKey(0),(dummy_imgs,dummy_labels))
  key=rnd.PRNGKey(0)

###########################################################################
# train loop

losses=[]
accuracies=[]
j=0

for epoch in range(num_epochs):
  start_time = time.time()
  for i, (x, y) in enumerate(loader):
    j+=1
    if y.shape!=(batch_size,):
      continue

    y = one_hot(y, n_targets)

    if mode=='kfac':
      key=rnd.split(key)[0]
      params,optstate,aux=optimizer.step(params,optstate,key,batch=(x,y),global_step_int=j)
      loss_=float(aux['loss'])
    else:
      loss_,grad=valgradfn(params,x,y,prevgrad)
      updates,optstate=optimizer.update(grad,optstate)
      params=optax.apply_updates(params,updates)
      prevgrad=grad

    accuracy_=accuracy(params, x, y)

    print(loss_)

    with open(os.path.join(logdir,'loss.txt'),'a') as f:
      f.write(str(loss_)+'\n')

    with open(os.path.join(logdir,'accuracy.txt'),'a') as f:
      f.write(str(accuracy_)+'\n')
