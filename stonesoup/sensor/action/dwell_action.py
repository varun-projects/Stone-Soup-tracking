# -*- coding: utf-8 -*-
import datetime

import numpy as np

from stonesoup.base import Property
from stonesoup.types.angle import Angle
from stonesoup.types.base import Type
from stonesoup.types.state import State


class ChangeDwellAction(Type):
    value: Angle = Property()
    owner: object = Property()
    start_time: datetime.datetime = Property()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # end-time is dependent on the sensor, and is left for the sensor to calculate
        self.end_time = None


class DwellActionsGenerator(Type):
    dwell_centre: State = Property()
    rpm: float = Property()
    fov: Angle = Property()
    owner: object = Property()
    start_time: datetime.datetime = Property()
    end_time: datetime.datetime = Property()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    def initial_bearing(self):
        return self.dwell_centre.state_vector[0, 0]

    @property
    def rps(self):
        return self.rpm / 60

    @property
    def angle_delta(self):
        return Angle(self.duration.total_seconds() * self.rps * 2 * np.pi)

    @property
    def min_(self):
        return Angle(self.initial_bearing - self.angle_delta)

    def max_(self):
        return Angle(self.initial_bearing + self.angle_delta)

    def __contains__(self, item):

        if isinstance(item, ChangeDwellAction):
            item = item.value

        if isinstance(item, (float, int)):
            item = Angle(item)

        left, right = Angle(self.min_ - self.fov / 2), Angle(self.max_ + self.fov / 2)

        if left < right:
            if left <= item <= right:
                return True
            else:
                return False
        else:
            if Angle(np.radians(-180)) <= item <= left or right <= Angle(np.radians(180)):
                return True
            else:
                return False

    def __iter__(self, resolution: Angle = np.radians(1)) -> ChangeDwellAction:
        """Returns CHangeDwellAction types, where the value is a possible value of the [0, 0]
        element of the dwell centre's state vector."""
        current_bearing = self.min_
        while current_bearing <= self.max_:
            yield ChangeDwellAction(value=current_bearing, owner=self.owner,
                                    start_time=self.start_time)
            current_bearing += resolution
