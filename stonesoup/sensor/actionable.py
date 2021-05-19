# -*- coding: utf-8 -*-
import datetime

import numpy as np

from stonesoup.base import Property
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.functions import mod_bearing
from stonesoup.sensor.action.dwell_action import DwellActionsGenerator, ChangeDwellAction
from stonesoup.sensor.radar import RadarBearingRange
from stonesoup.types.array import StateVector
from stonesoup.types.detection import TrueDetection
from stonesoup.types.state import State


class SimpleRadar(RadarBearingRange):
    dwell_centre: State = Property()
    rpm: float = Property()
    fov: float = Property()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._action = None

    def measurement_model(self, rot_offset=[0, 0, 0]):
        return CartesianToBearingRange(
            ndim_state=self.ndim_state,
            mapping=self.position_mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            rotation_offset=rot_offset)

    @property
    def default_action(self):
        """Default dwell action, used when input action is finished or when no inout action is
        given."""
        return ChangeDwellAction(value=self.dwell_centre.state_vector[0, 0],
                                 owner=self,
                                 start_time=self.dwell_centre.timestamp,
                                 end_time=None,
                                 increasing_angle=None,
                                 fov=self.fov)

    @property
    def rps(self):
        return self.rpm / 60

    def measure(self, ground_truths, noise, **kwargs):
        """Assumes that """

        if not ground_truths:
            return set()

        antenna_heading = self.orientation[2, 0] + self.dwell_centre.state_vector[0, 0]

        rot_offset = StateVector(
            [[self.orientation[0, 0]], [self.orientation[1, 0]], [antenna_heading]])

        measurement_model = CartesianToBearingRange(
            ndim_state=self.ndim_state,
            mapping=self.position_mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            rotation_offset=rot_offset)

        detections = set()
        for truth in ground_truths:
            # Transform state to measurement space and generate
            # random noise
            measurement_vector = measurement_model.function(truth, noise=noise, **kwargs)

            if noise is True:
                measurement_noise = measurement_model.rvs()
            else:
                measurement_noise = noise

            # Check if state falls within sensor's FOV
            fov_min = -self.fov / 2
            fov_max = +self.fov / 2
            bearing_t = measurement_vector[0, 0]

            # Do not measure if state not in FOV
            if not fov_min < bearing_t < fov_max:
                continue

            # Else add measurement
            measurement_vector += measurement_noise  # Add noise

            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)
            detections.add(detection)
        return detections

    def add_action(self, action):
        """Change current action to a given one."""
        if action.start_time != self.dwell_centre.timestamp:
            # need to think about this more, as sensor manager will need time to return action
            raise ValueError("Cannot schedule action that starts before current time.")

        self._action = action

    @property
    def current_action(self):
        """Returns the current action."""
        if self._action is None or not (
                self._action.start_time <= self.dwell_centre.timestamp < self._action.end_time):
            return self.default_action
        else:
            return self._action

    def _do_single_action(self, duration):
        """Do single action for a duration (assumes duration doesn't go beyond action end-time)."""
        angle_delta = duration.total_seconds() * self.rps * 2 * np.pi
        increasing = self.current_action.increasing_angle

        if increasing is None:
            # no direction, do nothing
            pass
        elif increasing:
            self.dwell_centre.state_vector[0, 0] = mod_bearing(self.dwell_centre.state_vector[0, 0]
                                                               + angle_delta)
        else:
            self.dwell_centre.state_vector[0, 0] = mod_bearing(self.dwell_centre.state_vector[0, 0]
                                                               - angle_delta)
        self.dwell_centre.timestamp += duration

    def do_action(self, timestamp):
        """Assumes only possible action is ChangeDwellAction type."""

        duration = timestamp - self.dwell_centre.timestamp

        if self.current_action.end_time is not None:
            # then action is not default, so has end-time

            if self.dwell_centre.timestamp + duration < self._action.end_time:
                # do current action until timestamp is reached
                action_duration = duration
            else:
                # do the rest of the action, and timestamp won't be reached
                action_duration = self.current_action.end_time - self.dwell_centre.timestamp

            self._do_single_action(action_duration)
            duration -= action_duration  # get remaining time

        # do default action until timestamp reached (duration might be 0)
        self._do_single_action(duration)

    def get_actions(self, timestamp: datetime.datetime) -> DwellActionsGenerator:
        """
        Method to create an action generator, concerning how the dwell centre can be modified
        at a particular timestamp.
        Assumes the dwell centre is up-to-date (ie. that 'now' for the sensor manager is the
        timestamp of the dwell centre's state).
        """

        return DwellActionsGenerator(dwell_centre=self.dwell_centre, rpm=self.rpm, fov=self.fov,
                                     owner=self,
                                     start_time=self.dwell_centre.timestamp, end_time=timestamp)
