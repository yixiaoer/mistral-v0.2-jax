from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import subprocess

from jax import Array
import jax.numpy as jnp

from mistral.lib.einshard import einshard

def set_device_count(n: int) -> None:
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + f' --xla_force_host_platform_device_count={n}'

def get_shard_shape(a: Array) -> tuple[int, ...]:
    return a.addressable_data(0).shape

def assert_equal(a, b):
    if a != b:
        raise AssertionError(f'{a} != {b}')

tests = [
    {
        'n_devices': 1,
        'shape': (2,),
        'expr': '... -> * ...',
        'ans': (2,),
    },
    {
        'n_devices': 16,
        'shape': (2,),
        'expr': 'a -> 16 a',
        'ans': (2,),
    },
    {
        'n_devices': 16,
        'shape': (2, 8),
        'expr': 'a b -> 2 a b*',
        'ans': (2, 1),
    },
    {
        'n_devices': 8,
        'shape': (4, 4),
        'expr': 'a b -> * a* b*',
        'ans': (2, 2),
    },
    {
        'n_devices': 16,
        'shape': (2, 8),
        'expr': '... a -> 2 ... a*',
        'ans': (2, 1),
    },
    {
        'n_devices': 16,
        'shape': (4, 16, 8),
        'expr': 'a b c -> a2* b2 c*',
        'ans': (1, 8, 4),
    },
]

def invoke_test(spec) -> None:
    set_device_count(spec['n_devices'])
    a = jnp.zeros(spec['shape'])
    a = einshard(a, spec['expr'])
    assert_equal(get_shard_shape(a), spec['ans'])

def main() -> None:
    if len(sys.argv) < 2:  # no command-line arguments provided
        for i in range(len(tests)):
            result = subprocess.call([sys.executable, __file__, f'{i}'])
            if result != 0:
                print(f'❌ Test {i} failed')
            else:
                print(f'✅ Test {i} passed')
    else:
        i = int(sys.argv[1])
        invoke_test(tests[i])

if __name__ == "__main__":
    main()
