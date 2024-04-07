from typing import NoReturn

class VoidType:
    '''A private class to implement the singleton pattern.'''
    def __repr__(self):
        return 'Void'

Void = VoidType()

def _void_type_new(cls) -> NoReturn:
    raise RuntimeError('VoidType cannot be instantiated directly')

def _void_type_init_subclass(cls, **kwargs) -> NoReturn:
    raise RuntimeError('Subclassing VoidType is not allowed')

VoidType.__new__ = _void_type_new
VoidType.__init_subclass__ = _void_type_init_subclass
