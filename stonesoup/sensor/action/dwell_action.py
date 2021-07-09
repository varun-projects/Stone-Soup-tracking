# -*- coding: utf-8 -*-
import datetime
from typing import Iterator

import numpy as np

from . import Action, ActionGenerator
from ...base import Property
from ...types.angle import Angle
from ...types.state import State


class ChangeDwellAction(Action):
    value: Angle = Property(readonly=True)
    owner: object = Property(readonly=True)
    start_time: datetime.datetime = Property(readonly=True)
    end_time: datetime.datetime = Property(readonly=True)
    increasing_angle: bool = Property(readonly=True)
    fov: Angle = Property(readonly=True)

    def __contains__(self, item):
        fov_min = self.value-self.fov / 2
        fov_max = self.value+self.fov / 2

        if isinstance(item, ChangeDwellAction):
            item = item.value

        if isinstance(item, (float, int)):
            item = Angle(item)

        if fov_min < fov_max:
            if fov_min <= item <= fov_max:
                return True
            else:
                return False
        else:
            if Angle(np.radians(-180)) <= item <= fov_min \
                    or fov_max <= item <= Angle(np.radians(180)):
                return True
            else:
                return False


class DwellActionsGenerator(ActionGenerator):
    dwell_centre: State = Property()
    rpm: float = Property()
    fov: Angle = Property()
    owner: object = Property()
    start_time: datetime.datetime = Property()
    end_time: datetime.datetime = Property()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution = Angle(np.radians(30))

    def __call__(self, resolution=None):
        if resolution is not None:
            self.resolution = resolution

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
    def min(self):
        if self.angle_delta >= np.pi:
            return Angle(-np.pi)
        else:
            return Angle(self.initial_bearing - self.angle_delta)

    @property
    def max(self):
        if self.angle_delta >= np.pi:
            return Angle(np.pi)
        else:
            return Angle(self.initial_bearing + self.angle_delta)

    def __contains__(self, item):

        if isinstance(item, ChangeDwellAction):
            item = item.value

        if isinstance(item, (float, int)):
            item = Angle(item)

        left, right = Angle(self.min - self.fov / 2), Angle(self.max + self.fov / 2)

        if left < right:
            if left <= item <= right:
                return True
            else:
                return False

        else:
            # return Angle(-np.pi) <= item <= left or right <= item <= Angle(np.pi)
            if Angle(np.radians(-180)) <= item <= left \
                    or right <= item <= Angle(np.radians(180)):
                return True
            else:
                return False

    def _get_end_time_direction(self, bearing):
        if self.initial_bearing <= bearing:
            if bearing - self.initial_bearing < self.initial_bearing + 2*np.pi - bearing:
                angle_delta = bearing - self.initial_bearing
                increasing = True
            else:
                angle_delta = self.initial_bearing + 2*np.pi - bearing
                increasing = False
        else:
            if self.initial_bearing - bearing < bearing + 2*np.pi - self.initial_bearing:
                angle_delta = self.initial_bearing - bearing
                increasing = False
            else:
                angle_delta = bearing + 2*np.pi - self.initial_bearing
                increasing = True

        return (
            self.start_time + datetime.timedelta(seconds=angle_delta / (self.rps*2*np.pi)),
            increasing
        )

    def __iter__(self) -> Iterator[ChangeDwellAction]:
        """Returns ChangeDwellAction types, where the value is a possible value of the [0, 0]
        element of the dwell centre's state vector."""
        current_bearing = self.min
        while current_bearing <= self.max:
            end_time, increasing = self._get_end_time_direction(current_bearing)
            yield ChangeDwellAction(value=current_bearing, owner=self.owner,
                                    start_time=self.start_time, end_time=end_time,
                                    increasing_angle=increasing, fov=self.fov)
            current_bearing += self.resolution

    def action_from_value(self, value):

        if isinstance(value, (int, float)):
            value = Angle(value)
        if not isinstance(value, Angle):
            raise ValueError("Can only generate action from an Angle/float/int type")

        if value not in self:
            return None  # Should this raise an error?

        value -= value % self.resolution

        end_time, increasing = self._get_end_time_direction(value)

        return ChangeDwellAction(value=value, owner=self.owner, start_time=self.start_time,
                                 end_time=end_time, increasing_angle=increasing, fov=self.fov)
