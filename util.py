def tostring(T, nonestring=''):
  notleaf=lambda T: isinstance(T,list) or isinstance(T,tuple)
  indent=lambda block:'\n'.join([u"\u2588"+l for l in block.splitlines()])
  if notleaf(T):
    return '\n\n'.join([indent(tostring(t)) for t in T])
  else:
    if T is None:
      return nonestring
    else:
      return str(T.shape)
    
def ts(T):
  print(tostring(T))

if __name__=='__main__':
  import jax.numpy as jnp
  A=jnp.ones((10,5))
  print(tostring([((A,),A),(A,)]))
