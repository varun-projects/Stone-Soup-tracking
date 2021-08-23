# -*- coding: utf-8 -*-

import numpy as np
import random
from datetime import datetime

from ...base import Base
from ...types.state import State
from ...types.track import Track
from ...sensor.actionable import SimpleRadar
from ...sensor.action.dwell_action import ChangeDwellAction
from ...sensormanager import RandomSensorManager, BruteForceSensorManager


def test_random_choose_actions():
    time_start = datetime.now()

    sensor = {SimpleRadar(position_mapping=(0, 2),
                          noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                                                [0, 0.75 ** 2]]),
                          ndim_state=4,
                          rpm=60,
                          fov=np.radians(30),
                          dwell_centre=State([0.0], time_start),
                          resolution=np.radians(30))}

    sensor_manager = RandomSensorManager(sensor)

    chosen_action_config = sensor_manager.choose_actions(time_start)
    assert type(chosen_action_config) == dict

    for sensor, action in chosen_action_config.items():
        assert isinstance(sensor, SimpleRadar)
        assert isinstance(action[0], ChangeDwellAction)


def test_brute_force_choose_actions():
    time_start = datetime.now()

    sensors = {SimpleRadar(position_mapping=(0, 2),
                           noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                                                 [0, 0.75 ** 2]]),
                           ndim_state=4,
                           rpm=60,
                           fov=np.radians(30),
                           dwell_centre=State([0.0], time_start),
                           resolution=np.radians(30))}

    track = [Track(states=[State(state_vector=[[0]],
                                 timestamp=time_start)])]

    class RewardFunction(Base):

        def calculate_reward(self, config, tracks_list, metric_time):
            config_metric = random.randint(0, 100)
            return config_metric

    reward_function = RewardFunction()

    sensor_manager = BruteForceSensorManager(sensors, reward_function.calculate_reward)

    chosen_action_config = sensor_manager.choose_actions(track, time_start)

    for chosen_action in chosen_action_config:
        for sensor, action in chosen_action.items():
            assert isinstance(sensor, SimpleRadar)
            assert isinstance(action[0], ChangeDwellAction)
