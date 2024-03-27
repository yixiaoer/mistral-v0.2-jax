from typing import Generic, TypeVar

T = TypeVar('T')
E = TypeVar('E')

class Either(Generic[T, E]):
    def is_success(self) -> bool:
        raise NotImplementedError('Subclasses should implement this method')

class success(Either[T, E]):
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

class fail(Either[T, E]):
    def __init__(self, error: E):
        self.error = error

    def __repr__(self):
        return f'fail({self.error!r})'

    def is_success(self) -> bool:
        return False

    def get(self) -> T:
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
