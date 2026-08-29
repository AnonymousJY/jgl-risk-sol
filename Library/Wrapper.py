import copy
from typing import Generic, TypeVar


T = TypeVar('T')


class Wrapper(Generic[T]):
    """
    Doc string TBD
    """
    def __init__(self, obj: T):
        self.obj = obj

    def __getattr__(self, item):
        return getattr(self.obj, item)

    @property
    def obj(self) -> T:
        return self._obj

    @obj.setter
    def obj(self, value: T) -> None:
        self._obj = copy.deepcopy(value)
