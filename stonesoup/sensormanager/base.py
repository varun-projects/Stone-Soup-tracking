# -*- coding: utf-8 -*-

from abc import abstractmethod, ABC
from typing import Callable, Set
# from random import sample, shuffle
import numpy as np
import itertools as it

from ..base import Base, Property
from ..sensor.sensor import Sensor
from ..predictor.kalman import KalmanPredictor
from ..models.measurement.nonlinear import CartesianToBearingRange


class SensorManager(Base, ABC):
    """The sensor manager base class.

    The purpose of a sensor manager is to return a set of sensor actions appropriate to a specific
    scenario and with a particular objective, or objectives, in mind. This involves using
    estimates of the situation and knowledge of the sensor system to calculate metrics associated
    with actions, and then determine optimal, or near optimal, actions to take.

    There is considerable freedom in both the theory and practice of sensor management and these
    classes do not enforce a particular solution. A sensor manager may be 'centralised' in that
    it controls the actions of multiple sensors, or individual sensors may have their own managers
    which communicate with other sensor managers in a networked fashion.

    """
    sensors: Set[Sensor] = Property(doc="The sensor(s) which the sensor manager is managing. "
                                        "These must be capable of returning available actions.")

    reward_function: Callable = Property(
        default=None, doc="A function or class designed to work out the reward associated with an "
                          "action or set of actions. This may also incorporate a notion of the "
                          "cost of making a measurement. The values returned may be scalar or "
                          "vector in the case of multi-objective optimisation. Metrics may be of "
                          "any type and in any units.")

    @abstractmethod
    def choose_actions(self, *args, **kwargs):
        """A method which returns a set of actions, designed to be enacted by a sensor, or
        sensors, chosen by some means. This will likely make use of optimisation algorithms.

        Returns
        -------
        : dict {Sensor: [Action]}
            Key-value pairs of the form 'sensor: actions'. In the general case a sensor may be
            given a single action, or a list. The actions themselves are objects which must be
            interpretable by the sensor to which they are assigned.
        """
        raise NotImplementedError


class RandomSensorManager(SensorManager):
    """As the name suggests, a sensor manager which returns a random choice of action or actions
    from the list available. Its practical purpose is to serve as a baseline to test against.

    """

    sensors: Set[Sensor] = Property(doc="The sensor(s) which the sensor manager is managing. "
                                        "These must be capable of returning available actions.")

    def choose_actions(self, timestamp, nchoose=1, *args, **kwargs):
        """Returns a randomly chosen [list of] action(s) from the action set for each sensor.

        Parameters
        ----------
        timestamp: :class:`datetime.datetime`
            Time at which the actions are being chosen

        nchoose : int
            Number of actions from the set to choose (default is 1)

        Returns
        -------
        : dict
            The pairs of {sensor: action(s) selected}
        """
        sensor_action_assignment = dict()

        for sensor in self.sensors:
            actions = sensor.get_actions(timestamp)
            sensor_action_assignment[sensor] = np.random.choice(list(actions))

        return sensor_action_assignment


class BruteForceSensorManager(SensorManager):
    """A sensor manager which returns a choice of action from those available,
    selecting the option which returns the maximum reward as calculated by a reward function.

    """

    sensors: Set[Sensor] = Property(doc="The sensor(s) which the sensor manager is managing. "
                                        "These must be capable of returning available actions.")
    predictor: KalmanPredictor = Property(doc="Predictor used to predict the track to a new state")
    reward_function: Callable = Property(doc="A function or class to calculate the reward "
                                             "associated with a given configuration of sensors "
                                             "and actions. The configuration which gives the "
                                             "maximum value of this reward will be selected as "
                                             "the chosen configuration of sensors and actions "
                                             "at this time step.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Generate a dictionary measurement models for each sensor
        # (for use before any measurements have been made)
        self.measurement_models = dict()
        for sensor in self.sensors:
            measurement_model = CartesianToBearingRange(
                ndim_state=4,
                mapping=(0, 2),
                noise_covar=sensor.noise_covar,
                translation_offset=sensor.position)
            self.measurement_models[sensor] = measurement_model

    def choose_actions(self, tracks_list, timestamp, nchoose=1, *args, **kwargs):
        """Returns a chosen [list of] action(s) from the action set for each sensor.
        Chosen action(s) is selected by finding the configuration of sensors: actions which returns
        the maximum reward, as calculated by a reward function.

        Parameters
        ----------
        tracks_list: list of :class:`~Track`
            List of tracks at given time. Used in reward function.
        timestamp: :class:`datetime.datetime`
            Time at which the actions are being chosen
        nchoose : int
            Number of actions from the set to choose (default is 1)

        Returns
        -------
        : dict
            The pairs of {sensor: action(s) selected}"""

        all_action_choices = dict()

        # For each sensor, randomly select an action to take
        for sensor in self.sensors:
            action_choices = list()
            actions = sensor.get_actions(timestamp)  # iterable
            for action in actions:
                action_choices.append(action)
                print(action)

            all_action_choices[sensor] = action_choices

        configs = ({sensor: action
                    for sensor, action in zip(all_action_choices.keys(), actionconfig)}
                   for actionconfig in it.product(*all_action_choices.values()))

        best_reward = -np.inf
        selected_config = None
        for config in configs:
            reward = self.reward_function(config, tracks_list, timestamp)
            if reward > best_reward:
                selected_config = config
                best_reward = reward

        # Return mapping of sensors and chosen actions for sensors
        return selected_config
