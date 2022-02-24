from typing import Iterable

from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.feeder import Feeder


class SimpleFeeder(Feeder):

    reader: Iterable = Property(doc="Source of states")

    @BufferedGenerator.generator_method
    def data_gen(self):
        for item in self.reader:
            yield item
