def tostring(T, nonestring=''):
  notleaf=lambda T: isinstance(T,list) or isinstance(T,tuple)
  indent=lambda block:'\n'.join(['| '+l for l in block.splitlines()])
  if notleaf(T):
    return 'O'+'\n'+'\n'.join([indent(tostring(t)) for t in T])
  else:
    if T is None:
      return nonestring
    else:
      return str(T.shape)

def testtostring(): 
  import jax.numpy as jnp
  A=jnp.ones((10,5))
  print(tostring([((A,),A),(A,)]))
