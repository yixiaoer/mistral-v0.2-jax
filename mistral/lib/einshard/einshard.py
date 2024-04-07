from math import prod

import jax
from jax import Array
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .parser import parse_expression

def _partition_at_ellipsis(lst: list) -> tuple[list, list]:
    idx = lst.index(...)
    l = lst[:idx]
    r = lst[idx + 1:]
    return l, r

def einshard(arr: Array, expression: str) -> Array:
    """
    Shards a :class:`jax.Array` according to a specified pattern, using a human-readable expression similar to that used in einsum notation.

    Args:
        arr (jax.Array): The Array to be processed with tensor parallelism.
        expression (str): A human-readable expression similar to einsum notation that specifies the sharding pattern.

    Returns:
        Array: The sharded array.
    """
    n_devices = jax.device_count()

    res = parse_expression(expression, 0)
    if not res.is_success():
        idx, desc = res.error
        raise ValueError(f'Cannot parse einshard expression "{expression}", expected {desc} at position {idx}.')
    _, (elements_left, elements_right) = res.value

    n_left_ellipses = sum(element_left is ... for element_left in elements_left)
    n_right_ellipses = sum(element_right is ... for element_right in elements_right)
    assert n_left_ellipses == n_right_ellipses and n_left_ellipses <= 1

    if n_left_ellipses > 0:  # == 1
        n_dims = len(arr.shape)
        n_dims_elided = n_dims - len(elements_left) + 1
        axis_names_for_left_augmented = [f'?{i}' for i in range(n_dims_elided)]
        axis_names_for_right_augmented = [(identifier, 1, False) for identifier in axis_names_for_left_augmented]  # 1: `sharding_number`, False: `is_proportional`

        elements_left_left, elements_left_right = _partition_at_ellipsis(elements_left)
        elements_left = [*elements_left_left, *axis_names_for_left_augmented, *elements_left_right]

        elements_right_left, elements_right_right = _partition_at_ellipsis(elements_right)
        elements_right = [*elements_right_left, *axis_names_for_right_augmented, *elements_right_right]

        # print(elements_left)
        # print(elements_right)

    sharding_numbers_fixed = [sharding_number for _, sharding_number, is_proportional in elements_right if not is_proportional]
    sharding_numbers_proportional = [sharding_number for _, sharding_number, is_proportional in elements_right if is_proportional]

    if not sharding_numbers_proportional:
        sharding_ratio = 1  # can be of whatever value because it will not be used in this case
    else:
        n_devices_needed_for_fixed = prod(sharding_numbers_fixed)
        n_devices_needed_for_proportional_base = prod(sharding_numbers_proportional)
        n_sharded_axes_proportional = len(sharding_numbers_proportional)

        assert n_devices % n_devices_needed_for_fixed == 0
        n_devices_available_proportional = n_devices // n_devices_needed_for_fixed

        assert n_devices_available_proportional % n_devices_needed_for_proportional_base == 0
        sharding_ratio_full = n_devices_available_proportional // n_devices_needed_for_proportional_base
        sharding_ratio = sharding_ratio_full ** (1 / n_sharded_axes_proportional)
        assert sharding_ratio.is_integer()
        sharding_ratio = int(sharding_ratio)

    mesh_shape = [sharding_number * (1 if not is_proportional else sharding_ratio) for _, sharding_number, is_proportional in elements_right]
    axis_names = tuple(f'a{i}' for i, _ in enumerate(elements_right))
    d = {identifier: i for i, (identifier, _, _) in enumerate(elements_right) if identifier is not None}
    partition_spec = tuple(f'a{d[element_left]}' for element_left in elements_left)

    # print(mesh_shape)
    # print(axis_names)
    # print(partition_spec)

    devices = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices, axis_names=axis_names)
    arr = jax.make_array_from_callback(arr.shape, NamedSharding(mesh, P(*partition_spec)), lambda idx: arr[idx])
    return arr
