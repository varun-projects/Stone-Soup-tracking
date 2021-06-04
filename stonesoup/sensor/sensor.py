import datetime
from abc import abstractmethod, ABC
from typing import Set, Union, Sequence

import numpy as np

from .base import PlatformMountable
from .action import Action, ActionGenerator
from ..types.detection import TrueDetection
from ..types.groundtruth import GroundTruthState


class Sensor(PlatformMountable, ABC):
    """Sensor Base class for general use.

    Most properties and methods are inherited from :class:`~.PlatformMountable`.

    Sensors must have a measure function.
    """

    @abstractmethod
    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:
        """Generate a measurement for a given state

        Parameters
        ----------
        ground_truths : Set[:class:`~.GroundTruthState`]
            A set of :class:`~.GroundTruthState`
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is `True`, in which
            case :meth:`~.Model.rvs` is used; if `False`, no noise will be added)

        Returns
        -------
        Set[:class:`~.TrueDetection`]
            A set of measurements generated from the given states. The timestamps of the
            measurements are set equal to that of the corresponding states that they were
            calculated from. Each measurement stores the ground truth path that it was produced
            from.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def measurement_model(self):
        """Measurement model of the sensor, describing general sensor model properties"""
        raise NotImplementedError

    def actions(self, timestamp: datetime.datetime, start_timestamp: datetime.datetime = None
                ) -> Set[ActionGenerator]:
        """Method to return a set of action generators available up to a provided timestamp.

        Parameters
        ----------
        timestamp: datetime.datetime
            Time that action would take place.
        start_timestamp: datetime.datetime, optional
            Time that start of action could take place.

        Returns
        -------
        : set of :class:`~.ActionGenerator`
            Set of action generators, that describe the bounds of each action space.
        """
        return set()

    def add_actions(self, actions: Sequence[Action]) -> bool:
        """Add actions to the sensor

        Parameters
        ----------
        actions: sequence of :class:`~.Action`
            Sequence of actions that will be executed in order

        Returns
        -------
        bool
            Return True if actions accepted. False if rejected.

        Raises
        ------
        NotImplementedError
            If sensor cannot be tasked.
        """
        raise NotImplementedError("Sensor cannot be tasked")

    def act(self, timestamp: datetime.datetime):
        """Carry out actions at timestamp.

        Parameters
        ----------
        timestamp: datetime.datetime
            Carry out actions up to this timestamp.
        """
        pass
