from math import prod
from types import EllipsisType
from typing import Any, Callable
import unicodedata

import jax
from jax import Array
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

ElementLeft = str | EllipsisType
ElementRight = tuple[str | None, int] | EllipsisType
Expression = tuple[list[ElementLeft], list[ElementRight]]

class ParseError(Exception):
    pass

is_identifier_char = lambda c: unicodedata.category(c)[0] == 'L' or c == '_'
is_0_to_9 = lambda c: c.isdigit()
is_1_to_9 = lambda c: c.isdigit() and c != '0'
is_space = lambda c: c.isspace()

def satisfy(predicate: Callable, description: str) -> Callable:
    def f(s: str, idx: int) -> tuple[int, str]:
        if idx == len(s):
            raise ParseError(f'Excepted {description} at position {idx}')
        c = s[idx]
        if not predicate(c):
            raise ParseError(f'Excepted {description} at position {idx}')
        idx += 1
        return idx, c
    return f

def many(parse: Callable) -> Callable:
    def f(s: str, idx: int) -> tuple[int, list[str]]:
        out = []
        try:
            while True:
                idx, token = parse(s, idx)
                out.append(token)
        except ParseError:
            pass
        return idx, out
    return f

def many1(parse: Callable) -> Callable:
    def f(s: str, idx: int) -> tuple[int, list[str]]:
        out = []
        idx, token = parse(s, idx)
        out.append(token)
        idx, tokens = many(parse)(s, idx)
        out.extend(tokens)
        return idx, out
    return f

def literal(s: str) -> Callable:
    len_literal = len(s)
    def f(s_: str, idx: int) -> tuple[int, str]:
        if idx > len(s_) - len_literal or s_[idx:idx+len_literal] != s:
            raise ParseError(f"Excepted string '{s}' at position {idx}")
        idx += len_literal
        return idx, s
    return f

def with_default(parse: Callable, *, default: Any) -> Callable:
    def f(s: str, idx: int) -> tuple[int, str]:
        try:
            idx, token = parse(s, idx)
        except ParseError:
            token = default
        return idx, token
    return f

parse_0_to_9 = satisfy(is_0_to_9, 'digit 0-9')
parse_1_to_9 = satisfy(is_1_to_9, 'digit 1-9')
parse_identifier_char = satisfy(is_identifier_char, 'identifier')
parse_space = satisfy(is_space, 'space')
parse_spaces_optional = many(parse_space)
parse_spaces = many1(parse_space)
parse_right_arrow = literal('->')
parse_ellipsis = literal('...')

def parse_identifier(s: str, idx: int) -> tuple[int, str]:
    idx, identifier_chars = many1(parse_identifier_char)(s, idx)
    return idx, ''.join(identifier_chars)

def parse_integer(s: str, idx: int) -> tuple[int, int]:
    out = []
    idx, digit = parse_1_to_9(s, idx)
    out.append(digit)
    try:
        while True:
            idx, digit = parse_0_to_9(s, idx)
            out.append(digit)
    except ParseError:
        pass
    return idx, int(''.join(out))

def parse_element_left(s: str, idx: int) -> tuple[int, ElementLeft]:
    try:
        return parse_identifier(s, idx)
    except ParseError:
        pass

    idx, _ = parse_ellipsis(s, idx)
    return idx, ...

def parse_element_right_without_ellipsis(s: str, idx: int) -> tuple[int, ElementRight]:
    idx_ = idx
    idx, identifier = with_default(parse_identifier, default=None)(s, idx)
    idx, integer = with_default(parse_integer, default=0)(s, idx)
    if idx_ == idx:
        raise ParseError()  # should have advancement
    return idx, (identifier, integer)

def parse_element_right(s: str, idx: int) -> tuple[int, ElementRight]:
    try:
        return parse_element_right_without_ellipsis(s, idx)
    except ParseError:
        pass
    idx, _ = parse_ellipsis(s, idx)
    return idx, ...

def parse_eof(s: str, idx: int) -> tuple[int, None]:
    if idx != len(s):
        raise ParseError(f'Excepted eof at position {idx}')
    return idx, None

def parse_expression(s: str, idx: int) -> tuple[int, Expression]:
    idx, _ = parse_spaces_optional(s, idx)

    elements_left = []
    try:
        while True:
            idx, identifier = parse_element_left(s, idx)
            elements_left.append(identifier)

            idx, _ = parse_spaces(s, idx)
    except ParseError:
        pass

    idx, _ = parse_right_arrow(s, idx)
    idx, _ = parse_spaces_optional(s, idx)

    elements_right = []
    try:
        while True:
            idx, element_right = parse_element_right(s, idx)
            elements_right.append(element_right)

            idx, _ = parse_spaces(s, idx)
    except ParseError:
        pass

    idx, _ = parse_eof(s, idx)
    return idx, (elements_left, elements_right)

# print(parse_element_left('a', 0))
# print(parse_element_left('...', 0))
# print(parse_element_right('...', 0))
# print(parse_expression('a ... b -> b ... a1', 0))

def partition_at_ellipsis(lst: list) -> tuple[list, list]:
    idx = lst.index(...)
    l = lst[:idx]
    r = lst[idx + 1:]
    return l, r

def einshard(arr: Array, expression: str) -> Array:
    n_devices = jax.device_count()

    _, (elements_left, elements_right) = parse_expression(expression, 0)

    n_left_ellipses = sum(element_left is ... for element_left in elements_left)
    n_right_ellipses = sum(element_right is ... for element_right in elements_right)
    assert n_left_ellipses == n_right_ellipses and n_left_ellipses <= 1

    if n_left_ellipses > 0:  # == 1
        n_dims = len(arr.shape)
        n_dims_elided = n_dims - len(elements_left) + 1
        axis_names_for_left_augmented = [f'?{i}' for i in range(n_dims_elided)]
        axis_names_for_right_augmented = [(item, 0) for item in axis_names_for_left_augmented]

        elements_left_left, elements_left_right = partition_at_ellipsis(elements_left)
        elements_left = [*elements_left_left, *axis_names_for_left_augmented, *elements_left_right]

        elements_right_left, elements_right_right = partition_at_ellipsis(elements_right)
        elements_right = [*elements_right_left, *axis_names_for_right_augmented, *elements_right_right]

        # print(elements_left)
        # print(elements_right)

    sharding_numbers = [integer for _, integer in elements_right if integer != 0]
    n_devices_base = prod(sharding_numbers)
    n_sharded_axes = len(sharding_numbers)
    assert n_devices % n_devices_base == 0
    sharding_ratio_full = n_devices // n_devices_base
    sharding_ratio_one = sharding_ratio_full ** (1 / n_sharded_axes)
    assert sharding_ratio_one.is_integer()
    sharding_ratio_one = int(sharding_ratio_one)

    mesh_shape = [1 if integer == 0 else integer * sharding_ratio_one for _, integer in elements_right]
    axis_names = tuple(f'a{i}' for i, _ in enumerate(elements_right))
    d = {identifier: i for i, (identifier, _) in enumerate(elements_right) if identifier is not None}
    partition_spec = tuple(f'a{d[element_left]}' for element_left in elements_left)

    # print('===========')
    # print(mesh_shape)
    # print(axis_names)
    # print(partition_spec)

    # print('----------')

    devices = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices, axis_names=axis_names)
    # print(mesh)
    # arr = jax.device_put(arr, NamedSharding(mesh, P(*partition_spec)))
    arr = jax.make_array_from_callback(arr.shape, NamedSharding(mesh, P(*partition_spec)), lambda idx: arr[idx])
    return arr

# print(jax.device_count())
# a = jnp.arange(32*4).reshape(32, 4)
# jax.debug.visualize_array_sharding(a)
# a = einshard(a, '... -> 1 ...')
# jax.debug.visualize_array_sharding(a)

# b = jnp.arange(32*4).reshape(32, 4)
# jax.debug.visualize_array_sharding(b)
# b = einshard(b, 'x y -> x1 y')
# jax.debug.visualize_array_sharding(b)

# arr = jnp.arange(32*4).reshape(32, 4)
# n_devices = jax.device_count() 
# mesh_shape = [n_devices, 1]
# axis_names = ('a1', 'a2')
# partition_spec = ('a1', 'a2')
# devices = mesh_utils.create_device_mesh(mesh_shape)
# mesh = Mesh(devices, axis_names=axis_names)
# arr = jax.make_array_from_callback(arr.shape, NamedSharding(mesh, P(*partition_spec)), lambda idx: arr[idx])
# jax.debug.visualize_array_sharding(arr)
# print(arr.addressable_data(0).shape)
