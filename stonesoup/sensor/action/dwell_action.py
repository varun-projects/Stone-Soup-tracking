# -*- coding: utf-8 -*-
import datetime
from typing import Any

import numpy as np

from stonesoup.base import Property
from stonesoup.types.angle import Angle
from stonesoup.types.base import Type


class ChangeDwellAction(Type):
    value: Any = Property()
    attr_name: str = Property()
    owner: object = Property()
    timestamp: datetime.datetime = Property()


class DwellActionsGenerator(Type):
    min_: Angle = Property()
    max_: Angle = Property()
    attr_name: str = Property()
    owner: object = Property()
    timestamp: datetime.datetime = Property()

    def __contains__(self, item):

        if isinstance(item, ChangeDwellAction):
            item = item.value

        if self.min_ <= item <= self.max_:
            return True
        else:
            return False

    def __iter__(self, resolution: Angle = np.radians(1)) -> ChangeDwellAction:
        current_dwell = self.min_
        while current_dwell <= self.max_:
            yield ChangeDwellAction(attr_name=self.attr_name, value=current_dwell,
                                    timestamp=self.timestamp)
            current_dwell += resolution
