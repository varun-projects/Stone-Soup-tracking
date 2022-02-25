# -*- coding: utf-8 -*-
import pytest

from ..simple import SimpleFeeder
from ...types.state import StateMutableSequence


@pytest.mark.parametrize('iterable', (list, tuple, StateMutableSequence))
def test_simple_feeder(iterable):
    iterable = iterable([1, 2, 3, 4])

    feeder = SimpleFeeder(iterable)

    for i, eval_elem in enumerate(feeder):
        assert eval_elem == iterable[i]
