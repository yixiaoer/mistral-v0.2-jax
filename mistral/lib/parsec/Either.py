from typing import Generic, TypeVar

T = TypeVar('T')
E = TypeVar('E')

type Either[T, E] = success[T] | fail[E]

class success(Generic[T]):
    def __init__(self, value: T):
        self.value = value

    def __repr__(self):
        return f'success({self.value!r})'

    def is_success(self) -> bool:
        return True

    def get(self) -> T:
        return self.value

    def __eq__(self, other):
        if not isinstance(other, success):
            return False
        return self.value == other.value

class fail(Generic[E]):
    def __init__(self, error: E):
        self.error = error

    def __repr__(self):
        return f'fail({self.error!r})'

    def is_success(self) -> bool:
        return False

    def get(self) -> E:
        return self.error

    def __eq__(self, other):
        if not isinstance(other, fail):
            return False
        return self.error == other.error

# def might_fail(n: int) -> Either[str, int]:
#     if n > 0:
#         return Success('Positive number')
#     else:
#         return Fail(-1)

# result = might_fail(-1)
# print(result)
