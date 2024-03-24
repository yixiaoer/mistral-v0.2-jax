import jax.numpy as jnp
import numpy as np
import torch

# PyTorch -> NumPy -> JAX
# JAX -> NumPy -> PyTorch

def pt2np(arr):
    with torch.no_grad():
        return arr.cpu().numpy()

def np2jax(arr):
    return jnp.asarray(arr)

def pt2jax(arr):
    with torch.no_grad():
        return np2jax(pt2np(arr))

def jax2np(arr):
    return np.asarray(arr).copy()

def jax2np_noncopy(arr):
    return np.asarray(arr)

def np2pt(arr):
    return torch.from_numpy(arr)

def jax2pt(arr):
    return np2pt(jax2np(arr))

def jax2pt_noncopy(arr):
    return np2pt(jax2np_noncopy(arr))
