# -*- coding: utf-8 -*-

from abc import ABC

from ..base import Base

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

    @abstractmethod
    def choose_actions(self, *args, **kwargs):
        """A method which returns a set of actions, designed to be enacted by a sensor, or
        sensors, chosen by some means. This will likely make use of optimisation algorithms.

        Returns
        -------
        : Set
            A set of actions interpretable by the input set of sensors


        """

        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self, *args, **kwargs):
        """A method designed to work out the reward associated with an action or set of actions.
        This may also incorporate a notion of the cost of making a measurment. The values returned
        may be scalar or vector in the case of multi-objective optimisation. Metrics may be of any
        type and in any units.

        Returns
        -------
        : (various)
            A reward or rewards
        """
        raise NotImplementedError


class DiscreteSensorManager(SensorManager, ABC):
    """A sensor msnager in which the actions comprise a discrete collaction of objects. It is
    presumed that actions can refer to single actions or multiple actions across different
    sensors. Potentisl actions are unique and their order is not important.

    """
    action_set : set = Property(doc="")
