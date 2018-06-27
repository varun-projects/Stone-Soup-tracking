# -*- coding: utf-8 -*-
import numpy as np
import pytest

from ..prediction import Prediction, MeasurementPrediction
from ..detection import Detection
from ..track import Track
from ..hypothesis import (
    Hypothesis, DistanceHypothesis, JointHypothesis, DistanceJointHypothesis)

prediction = Prediction(np.array([[1], [0]]))
measurement_prediction = MeasurementPrediction(np.array([[1], [0]]))
detection = Detection(np.array([[1], [0]]))
distance = float(1)


def test_hypothesis():
    """Hypothesis type test"""

    with pytest.raises(TypeError):
        Hypothesis(prediction, measurement_prediction, detection)


def test_distance_hypothesis():
    """Distance Hypothesis type test"""

    hypothesis = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance)

    assert hypothesis.prediction is prediction
    assert hypothesis.measurement_prediction is measurement_prediction
    assert hypothesis.detection is detection
    assert hypothesis.distance is distance


def test_distance_hypothesis_comparison():
    """Distance Hypothesis comparison test"""

    h1 = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance)
    h2 = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance + 1)

    assert h1 > h2
    assert h2 < h1
    assert h1 <= h1
    assert h1 >= h1
    assert h1 == h1


def test_distance_joint_hypothesis():
    """Distance Joint Hypothesis type test"""

    t1 = Track()
    t2 = Track()
    h1 = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance)
    h2 = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance)

    hypotheses = {t1: h1, t2: h2}
    joint_hypothesis = JointHypothesis(hypotheses)

    assert isinstance(joint_hypothesis, DistanceJointHypothesis)
    assert joint_hypothesis[t1] is h1
    assert joint_hypothesis[t2] is h2
    assert joint_hypothesis.distance == distance * 2


def test_distance_joint_hypothesis_comparison():
    """Distance Joint Hypothesis comparison test"""

    t1 = Track()
    t2 = Track()
    h1 = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance)
    h2 = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance)
    h3 = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance + 1)

    hypotheses1 = {t1: h1, t2: h2}
    hypotheses2 = {t1: h1, t2: h3}
    j1 = JointHypothesis(hypotheses1)
    j2 = JointHypothesis(hypotheses2)

    assert j1 > j2
    assert j2 < j1
    assert j1 <= j1
    assert j1 >= j1
    assert j1 == j1


def test_invalid_joint_hypothesis():
    """Invalid Joint Hypothesis test"""

    t1 = Track()
    t2 = Track()

    h1 = object()
    h2 = object()

    hypotheses = {t1: h1, t2: h2}

    with pytest.raises(NotImplementedError):
        JointHypothesis(hypotheses)
