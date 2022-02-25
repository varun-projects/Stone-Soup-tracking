from typing import Iterable

from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..feeder import Feeder


class SimpleFeeder(Feeder):
    """Simple data feeder

    Creates a generator from an iterable.
    """
    reader: Iterable = Property(doc="Source of states")

    @BufferedGenerator.generator_method
    def data_gen(self):
        for item in self.reader:
            yield item
