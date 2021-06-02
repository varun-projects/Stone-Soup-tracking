from abc import abstractmethod
from typing import Collection, Iterator

from ...base import Base


class Action(Base):

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return all(getattr(self, name) == getattr(other, name) for name in type(self).properties)

    def __hash__(self):
        return hash(tuple(getattr(self, name) for name in type(self).properties))


class ActionGenerator(Base, Collection[Action]):

    @abstractmethod
    def __contains__(self, item):
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self) -> Iterator[Action]:
        raise NotImplementedError()
