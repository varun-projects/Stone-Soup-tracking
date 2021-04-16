# -*- coding: utf-8 -*-

from abc import abstractmethod, ABC
from typing import Callable, Set
from random import sample, shuffle

from ..base import Base, Property
from ..sensor.sensor import Sensor


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

    def choose_actions(self, nchoose=1, *args, **kwargs):
        """Return a randomly chosen [list of] action(s) from the action set. To ensure no
        order-preservation, the action set is first listified and then shuffled before a sample
        is selected.

        Parameters
        ----------
        nchoose : int
            Number of actions from the set to choose (default is 1)

        Returns
        -------
        : dict
            The pairs of {sensor: action(s) selected}
        """
        out_dict = dict()

        for sensor in self.sensors:
            out_dict[sensor] = sensor.action_set[sample(shuffle(list(sensor.action_set)),
                                                        k=nchoose)]

        return out_dict
