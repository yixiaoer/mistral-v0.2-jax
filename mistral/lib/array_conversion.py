from jax import Array
import jax.numpy as jnp
import numpy as np
import torch

# PyTorch -> NumPy -> JAX
# JAX -> NumPy -> PyTorch

def pt2np(arr: torch.tensor) -> np.ndarray:
    '''
    Converts a PyTorch array into a NumPy array.

    Args:
        x (torch.tensor): PyTorch array to convert.

    Returns:
        np.ndarray: Converted NumPy array.
    '''
    with torch.no_grad():
        return arr.cpu().numpy()

def np2jax(arr: np.ndarray) -> Array:
    '''
    Converts a NumPy array into a JAX array.

    Args:
        x (np.ndarray): NumPy array to convert.

    Returns:
        Array: Converted jax.Array.
    '''
    return jnp.asarray(arr)

def pt2jax(arr: torch.tensor) -> Array:
    '''
    Converts a PyTorch array into a JAX array. The process involves converting the PyTorch tensor to a NumPy array first, then to JAX array.

    Args:
        x (torch.tensor): PyTorch array to convert.

    Returns:
        Array: Converted jax.Array.
    '''
    with torch.no_grad():
        return np2jax(pt2np(arr))

def jax2np(arr: Array) -> np.ndarray:
    '''
    Converts a JAX array into a NumPy array.

    Args:
        x (Array): JAX array to convert.

    Returns:
        np.ndarray: Converted NumPy array.
    '''
    return np.asarray(arr).copy()

def jax2np_noncopy(arr: Array) -> np.ndarray:
    '''
    Converts a JAX array into a NumPy array. The conversion process tries to avoid unnecessary copying when possible.

    Args:
        x (Array): JAX array to convert.

    Returns:
        np.ndarray: Converted NumPy array.
    '''
    return np.asarray(arr)

def np2pt(arr: np.ndarray) -> torch.tensor:
    '''
    Converts a NumPy array into a PyTorch tensor.

    Args:
        x (np.ndarray): NumPy array to convert.

    Returns:
        torch.tensor: Converted tensor.
    '''
    return torch.from_numpy(arr)

def jax2pt(arr: Array) -> torch.tensor:
    '''
    Converts a JAX array into a PyTorch tensor.

    Args:
        x (Array): JAX array to convert.

    Returns:
        torch.tensor: Converted tensor.
    '''
    return np2pt(jax2np(arr))

def jax2pt_noncopy(arr: Array) -> torch.tensor:
    '''
    Converts a JAX array into a PyTorch tensor. The conversion process tries to avoid unnecessary copying when possible.

    Args:
        x (Array): JAX array to convert.

    Returns:
        torch.tensor: Converted tensor.
    '''
    return np2pt(jax2np_noncopy(arr))
