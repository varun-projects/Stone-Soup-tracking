#!/usr/bin/env python
# coding: utf-8

"""
==========================================================
3 - Target Based Sensor Management
==========================================================
"""

# %%
# 
# This tutorial uses the sensor manager classes in Stone Soup which have been demonstrated in the previous
# sensor management tutorials and further develops classes for target based sensor management.
# 
# Target based sensor management
# ------------------------------
#
# The example in this tutorial considers the same scenario as previous sensor management tutorials
# - 10 targets moving on nearly constant velocity trajectories and an adjustable number of
# :class:`~.SimpleRadar` sensors with a defined field of view which can be pointed in a
# particular direction in order to make an observation.
#
# The sensor management methods explored here are based on the sensor manager classes available in the
# stone soup framework but developed further to consider the estimated location of the targets before applying
# the sensor management algorithms. Each method first estimates the angles between the sensor(s) and each target's
# predicted location. The sensor management algorithms then use the sensor's :meth:`action_from_value` method to
# generate the action for the sensor to take which would points the dwell centre at the given angle.
# The sensor managers then select the action(s)
# for the sensor(s) using their respective methods.
#
# The first method, :class:`TargetBasedRandomManager` selects an action randomly from the possible actions,
# with equal probability.
# As in the previous tutorials the second method, :class:`TargetBasedBruteForceManager` aims to reduce the total
# uncertainty of the track estimates at each time step. To achieve this the sensor manager considers all
# combinations of the possible directions for the sensor(s) to point in. The sensor manager
# chooses the configuration for which the sum of estimated uncertainties (as represented by the Frobenius
# norm of the covariance matrix) can be reduced the most by observing the targets within the field of view
# when pointed in the chosen direction.
#
# Only considering actions which point the sensor in the direction of an estimated location for a target reduces the
# possibilities that the brute force method needs to consider. The possible actions for a sensor to take
# are no longer any direction available at the given resolution, just the directions where there is predicted
# to be a target. In this scenario the maximum number of actions to choose from is 10 per sensor - one for each target.
# This reduces the run time of the algorithm, even for a larger number of sensors.
#
# Sensor Management example
# -------------------------
# 
# Setup
# ^^^^^
# 
# First a simulation must be set up using components from Stone Soup. For this the following imports are required.

import numpy as np
import random
from datetime import datetime, timedelta

start_time = datetime.now()

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from stonesoup.base import Property, Base
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import Detection

# %%
# Generate ground truth
# ^^^^^^^^^^^^^^^^^^^^^
# 
# Generate transition model and ground truths as in Tutorials 1 and 2.
#
# The number of targets in this simulation is defined by `n_truths` - here there are 10 targets.
# The time the simulation is observed for is defined by `time_max`.
#
# We can fix our random number generator in order to probe a particular example repeatedly.
# This can be undone by commenting out the first two lines in the next cell.

np.random.seed(1990)
random.seed(1990)

# Generate transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])

yps = range(0, 100, 10)  # y value for prior state
truths = []
ntruths = 10  # number of ground truths in simulation
time_max = 100  # timestamps the simulation is observed over

# Generate ground truths
for j in range(0, ntruths):
    truth = GroundTruthPath([GroundTruthState([0, 1, yps[j], 1], timestamp=start_time)],
                            id=f"id{j}")

    for k in range(1, time_max):
        truth.append(
            GroundTruthState(transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
                             timestamp=start_time + timedelta(seconds=k)))
    truths.append(truth)

# %%
# Plot the ground truths. This is done using the :class:`~.Plotter` class from Stone Soup.

from stonesoup.plotter import Plotter

# Stonesoup plotter requires sets not lists
truths_set = set(truths)

plotter = Plotter()
plotter.ax.axis('auto')
plotter.plot_ground_truths(truths_set, [0, 2])

# %%
# Create sensors
# ^^^^^^^^^^^^^^
#
# Create a sensor for each sensor management algorithm. As in the previous tutorials a specifically developed sensor
# :class:`~.SimpleRadar` is used here.

total_no_sensors = 1

from stonesoup.types.state import State
from stonesoup.sensor.actionable import SimpleRadar

sensor_setA = set()

for n in range(0, total_no_sensors):
    sensor = SimpleRadar(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        ndim_state=4,
        position=np.array([[10], [n * 50]]),
        rpm=60,
        fov=np.radians(30),
        dwell_centre=State([0.0], start_time),
        resolution=np.radians(30)
    )
    sensor_setA.add(sensor)

sensor_setB = set()

for n in range(0, total_no_sensors):
    sensor = SimpleRadar(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        ndim_state=4,
        position=np.array([[10], [n * 50]]),
        rpm=60,
        fov=np.radians(30),
        dwell_centre=State([0.0], start_time),
        resolution=np.radians(30)
    )
    sensor_setB.add(sensor)
# %%
# Create the Kalman predictor and updater
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Construct a predictor and updater using the :class:`~.KalmanPredictor` and :class:`~.ExtendedKalmanUpdater`
# components from Stone Soup. The measurement model for the updater is `None` as it is an attribute of the sensor.

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import ExtendedKalmanUpdater
updater = ExtendedKalmanUpdater(measurement_model=None)
# measurement model is added to detections by the sensor

# %%
# Run the Kalman filters
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Create priors which estimate the targets’ initial states - these are the same as in the previous
# sensor management tutorials.

from stonesoup.types.state import GaussianState

priors = []
for j in range(0, ntruths):
    priors.append(
        GaussianState([[0], [1.5], [yps[j] + 0.5], [1.5]], np.diag([1.5, 0.25, 1.5, 0.25] + np.random.normal(0, 5e-4, 4)),
                      timestamp=start_time))

# %%
# Initialise the tracks by creating an empty list and appending the priors generated. This needs to be done separately
# for both sensor manager methods as they will generate different sets of tracks.

from stonesoup.types.track import Track

# Initialise tracks from the TargetBasedRandomManager
tracksA = []
for j, prior in enumerate(priors):
    tracksA.append(Track([prior], id=f"id{j}"))

# Initialise tracks from the TargetBasedBruteForceManager
tracksB = []
for j, prior in enumerate(priors):
    tracksB.append(Track([prior], id=f"id{j}"))

# %%
# Create sensor managers
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Next we create our sensor manager classes. Two sensor manager classes are created in this tutorial
# - :class:`TargetBasedRandomManager` and :class:`TargetBasedBruteForceManager`. They are based on
# the :class:`~.RandomSensorManager` and :class:`~.BruteForceSensorManager` classes from the stone soup framework.
# Both methods first calculate the angles from the sensor(s) to the estimated positions of the targets and then use
# their respective algorithms to choose an action for the sensor(s), selecting a direction to point in from
# those which will likely observe a target.
# 
# Target based random manager
# """""""""""""""""""""""""""
# 
# The first method :class:`TargetBasedRandomManager`, randomly chooses the action(s) for the sensor to take
# to make an observation. To do this the :meth:`choose_actions`
# function uses :meth:`random.sample()` to draw a random sample from all directions the sensor could point in
# which would directly observe a target.

from stonesoup.sensormanager import RandomSensorManager


class TargetBasedRandomManager(RandomSensorManager):
    predictor: KalmanPredictor = Property()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_actions(self, tracks_list, timestamp, nchoose=1):
        sensor_action_assignment = dict()

        # For each sensor, randomly select an action to take
        for sensor in self.sensors:
            action_generators = sensor.actions(timestamp)
            chosen_actions = []  # selected action(s) for sensor to take

            angles_to_targets = []
            for track in tracks_list:
                prediction = self.predictor.predict(track[-1], timestamp)
                relative_position = prediction.state_vector[[0, 2], :] - sensor.position[[0, 1], :]
                angle_to_target = np.arctan2(relative_position[1], relative_position[0])
                angles_to_targets.append(angle_to_target)

            for action_gen in action_generators:
                action_choices = []  # possible actions from action_gen
                for angle in angles_to_targets:
                    if angle in action_gen:
                        action_choices.append(action_gen.action_from_value(angle_to_target))

                chosen_actions.extend(random.sample(action_choices, k=nchoose))

            sensor_action_assignment[sensor] = chosen_actions

            # Return dictionary of sensors and actions to take
        return sensor_action_assignment

# %%
# Target based brute force manager
# """"""""""""""""""""""""""""""""
# 
# The second method :class:`TargetBasedBruteForceManager` iterates through the possible actions a sensor can take
# that result in the sensor pointing at a target and selects the action(s) which give the maximum
# reward as calculated by the reward function.
# In this example the reward function is written such that the sensor
# manager chooses a direction for the sensor to point in such that the total uncertainty of the tracks will be
# reduced the most by making an observation in that direction.

from stonesoup.sensormanager import BruteForceSensorManager
import itertools as it


class TargetBasedBruteForceManager(BruteForceSensorManager):
    predictor: KalmanPredictor = Property()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_actions(self, tracks_list, timestamp, nchoose=1):
        all_action_choices = dict()

        # For each sensor, randomly select an action to take
        for sensor in self.sensors:
            action_generators = sensor.actions(timestamp)
            sensor_action_choices = list(it.product(*action_generators))  # all possible actions
            selected_action_choices = []  # actions which point at targets

            angles_to_targets = []
            for track in tracks_list:
                prediction = self.predictor.predict(track[-1], timestamp)
                relative_position = prediction.state_vector[[0, 2], :] - sensor.position[[0, 1], :]
                angle_to_target = np.arctan2(relative_position[1], relative_position[0])
                angles_to_targets.append(angle_to_target)

            best_choice_metric = 0
            for choice in sensor_action_choices:
                choice_metric = 0
                for item in choice:
                    for angle in angles_to_targets:
                        if angle in item:
                            choice_metric += 1
                if choice_metric >= best_choice_metric:
                    best_choice_metric = choice_metric
                    selected_action_choices.append(choice)

            # dictionary of sensors: list(action combinations)
            all_action_choices[sensor] = selected_action_choices  # list of tuples

        # get tuple of dictionaries of sensors: actions
        configs = ({sensor: action
                    for sensor, action in zip(all_action_choices.keys(), actionconfig)}
                   for actionconfig in it.product(*all_action_choices.values()))

        best_rewards = np.zeros(nchoose) - np.inf
        selected_configs = [None] * nchoose
        for config in configs:
            # calculate reward for dictionary of sensors: actions
            reward = self.reward_function(config, tracks_list, timestamp)
            if reward > min(best_rewards):
                selected_configs[np.argmin(best_rewards)] = config
                best_rewards[np.argmin(best_rewards)] = reward

        # Return mapping of sensors and chosen actions for sensors
        return selected_configs


# %%
# Reward function
# """""""""""""""
# The :class:`RewardFunction` is the same as in the previous tutorials and calculates the difference between
# the covariance matrix norms of the
# prediction and the posterior assuming a predicted measurement corresponding to that prediction.

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange


class RewardFunction(Base):
    predictor: KalmanPredictor = Property(doc="Predictor used to predict the track to a new state")
    updater: ExtendedKalmanUpdater = Property(doc="Updater used in the reward function to update "
                                                  "the track to the new state.")

    def calculate_reward(self, config, tracks_list, metric_time):

        # Reward value
        config_metric = 0

        # Create dictionary of predictions for the tracks in the configuration
        predictions = {track: self.predictor.predict(track[-1],
                                                     timestamp=metric_time)
                       for track in tracks_list}

        # Running updates
        r_updates = dict()

        # For each sensor in the configuration
        for sensor, actions in config.items():

            measurement_model = CartesianToBearingRange(
                ndim_state=sensor.ndim_state,
                mapping=sensor.position_mapping,
                noise_covar=sensor.noise_covar,
                translation_offset=sensor.position)

            # Provide the updater with the correct measurement model for the sensor
            self.updater.measurement_model = measurement_model

            for track in tracks_list:
                predicted_measurement = self.updater.predict_measurement(predictions[track])
                angle_to_target = predicted_measurement.state_vector[0]

                for action in actions:
                    if angle_to_target in action:

                        # If the track is selected by a sensor for the first time 'previous' is the prediction
                        # If the track has already been selected by a sensor 'previous' is the most recent update
                        if track not in r_updates:
                            previous = predictions[track]
                        else:
                            previous = r_updates[track]

                        previous_cov_norm = np.linalg.norm(previous.covar)

                        # Calculate predicted measurement
                        predicted_measurement = self.updater.predict_measurement(previous)

                        # Generate detection from predicted measurement
                        detection = Detection(predicted_measurement.state_vector,
                                              timestamp=metric_time)

                        # Generate hypothesis based on prediction/previous update and detection
                        hypothesis = SingleHypothesis(previous, detection)

                        # Do the update based on this hypothesis and store covariance matrix
                        update = self.updater.update(hypothesis)
                        update_cov_norm = np.linalg.norm(update.covar)

                        # Replace prediction in dictionary with update
                        r_updates[track] = update

                        # Calculate metric for the track observation and add to the metric for the configuration
                        metric = previous_cov_norm - update_cov_norm
                        config_metric += metric

        # Return value of configuration metric
        return config_metric


# %%
# Initiate sensor managers
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create an instance of each sensor manager class. As in the previous tutorials each class takes in a
# set of `sensors`, and the :class:`TargetBasedBruteForceManager` takes in a callable reward function.
# Additionally, each sensor manager requires a predictor for estimating the angles to targets.

randomactionmanager = TargetBasedRandomManager(sensor_setA, predictor)

# initiate reward function
reward_function = RewardFunction(predictor, updater)

bruteforceactionmanager = TargetBasedBruteForceManager(sensors=sensor_setB,
                                                       reward_function=reward_function.calculate_reward,
                                                       predictor=predictor)

# %%
# Run the sensor managers
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# From here the method is the same as in the previous sensor management tutorials.
# 
# For both sensor management methods, at each time step the chosen action is given to the sensors
# and then measurements taken. The tracks are updated based on these measurements with predictions made
# for tracks which have not been observed.
# 
# First a hypothesiser and data associator are required for use in both trackers.
#

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=5)

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
data_associator = GNNWith2DAssignment(hypothesiser)

# %%
# Run target based random sensor manager
# """"""""""""""""""""""""""""""""""""""
#
# Here the chosen target for observation is selected randomly using the method :meth:`choose_actions()` from the class
# :class:`TargetBasedRandomManager`. This returns a mapping of sensors to actions where actions are a direction
# for the sensor to point in, selected randomly.

from ordered_set import OrderedSet

# Generate list of timesteps from ground truth timestamps
timesteps = []
for state in truths[0]:
    timesteps.append(state.timestamp)

for timestep in timesteps[1:]:

    # Generate chosen configuration
    chosen_action = randomactionmanager.choose_actions(tracksA, timestep)

    # Create empty dictionary for measurements
    measurementsA = []

    for sensor, actions in chosen_action.items():
        sensor.add_actions(actions)
    #         print(np.rad2deg(action.value))

    for sensor in sensor_setA:
        sensor.act(timestep)

        # Observe this ground truth
        measurements = sensor.measure(OrderedSet(truth[timestep] for truth in truths), noise=True)
        measurementsA.extend(measurements)

    hypotheses = data_associator.associate(tracksA,
                                           measurementsA,
                                           timestep)
    for track in tracksA:
        hypothesis = hypotheses[track]
        if hypothesis.measurement:
            post = updater.update(hypothesis)
            track.append(post)
        else:  # When data associator says no detections are good enough, we'll keep the prediction
            track.append(hypothesis.prediction)

# %%
# Plot ground truths, tracks and uncertainty ellipses for each target. The positions of the sensors are
# indicated by black x markers.

from matplotlib.lines import Line2D
from stonesoup.plotter import Plotter

plotterA = Plotter()
plotterA.ax.axis('auto')

# Plot sensor positions as black x markers
for sensor in sensor_setA:
    plotterA.ax.scatter(sensor.position[0], sensor.position[1], marker='x', c='black')
plotterA.labels_list.append('Sensor')
plotterA.handles_list.append(Line2D([], [], linestyle='', marker='x', c='black'))

plotterA.plot_ground_truths(truths_set, [0, 2])
plotterA.plot_tracks(set(tracksA), [0, 2], uncertainty=True)

# %%
# In comparison to the previous tutorials the performance of the random sensor mangement method has improved. For the
# same scenario and same number of sensors the tracks are better with smaller uncertainty. This is because the
# random choice is not chosen from any direction the sensor could possibly point in but from any direction that is
# estimated to point at a target. This improves the performance of the algorithm as an observation is much more likely.
#
# Run target based brute force sensor manager
# """""""""""""""""""""""""""""""""""""""""""
#
# Here the direction for observation is selected based on the difference between the covariance matrices of the
# prediction and the update of predicted measurement, for targets which could be observed by the sensor
# pointing in the given direction.
#
# Within the sensor manager a dictionary is created of sensors and the actions they can take which would result in
# the sensor pointing at a target.
# When the :meth:`choose_actions` function is called (at each time step), for each track in the tracks list:
#
# * A prediction is made and the covariance matrix norms stored
# * The angle from the sensor to the prediction is calculated to establish if it within the field of view
# * If it is in the field of view a predicted measurement is made
# * A synthetic detection is generated from this predicted measurement
# * A hypothesis generated based on the detection and prediction
# * This hypothesis is used to do an update and the covariance matrix norms of the update are stored.
#
# The metric `config_metric` is calculated as the sum of the differences between these covariance matrix norms
# for the tracks observed by the possible action configuration. The sensor manager identifies the
# configuration which results in the largest value of this metric and therefore
# largest reduction in uncertainty. It returns the optimum sensors/actions configuration as a dictionary.
#
# The actions are given to the sensors, measurements made and
# the tracks updated based on these measurements. Predictions are made for tracks
# which have not been observed by the sensors.

for timestep in timesteps[1:]:

    # Generate chosen configuration
    chosen_actions = bruteforceactionmanager.choose_actions(tracksB, timestep)

    # Create empty dictionary for measurements
    measurementsB = []

    for chosen_action in chosen_actions:
        for sensor, actions in chosen_action.items():
            sensor.add_actions(actions)

    for sensor in sensor_setB:
        sensor.act(timestep)

        # Observe this ground truth
        measurements = sensor.measure(OrderedSet(truth[timestep] for truth in truths), noise=True)
        measurementsB.extend(measurements)

    hypotheses = data_associator.associate(tracksB,
                                           measurementsB,
                                           timestep)
    for track in tracksB:
        hypothesis = hypotheses[track]
        if hypothesis.measurement:
            post = updater.update(hypothesis)
            track.append(post)
        else:  # When data associator says no detections are good enough, we'll keep the prediction
            track.append(hypothesis.prediction)

# %%
# Plot ground truths, tracks and uncertainty ellipses for each target.

plotterB = Plotter()
plotterB.ax.axis('auto')

# Plot sensor positions as black x markers
for sensor in sensor_setB:
    plotterB.ax.scatter(sensor.position[0], sensor.position[1], marker='x', c='black')
# Add to legend generated
plotterB.labels_list.append('Sensor')
plotterB.handles_list.append(Line2D([], [], linestyle='', marker='x', c='black'))

plotterB.plot_ground_truths(truths_set, [0, 2])
plotterB.plot_tracks(set(tracksB), [0, 2], uncertainty=True)

# %%
# It is clear from these initial plots that the brute force method produces better tracks than the random method.
# The smaller uncertainty ellipses suggest that the :class:`TargetBasedBruteForceManager` provides a much
# better track than the :class:`TargetBasedRandomManager`.
#
# Metrics
# -------
# 
# Metrics can be used to compare how well different sensor management techniques are working. Full explanations of
# the OSPA and SIAP metrics can be found in the Metrics Example.

from stonesoup.metricgenerator.ospametric import OSPAMetric
ospa_generator = OSPAMetric(c=40, p=1)

from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
siap_generator = SIAPMetrics(position_mapping=[0, 2], velocity_mapping=[1, 3])

from stonesoup.dataassociator.tracktotrack import TrackToTruth
associator = TrackToTruth(association_threshold=30)

from stonesoup.metricgenerator.uncertaintymetric import SumofCovarianceNormsMetric
uncertainty_generator = SumofCovarianceNormsMetric()

# %%
# Generate a metrics manager for each sensor management method.

from stonesoup.metricgenerator.manager import SimpleManager

metric_managerA = SimpleManager([ospa_generator, siap_generator, uncertainty_generator],
                                associator=associator)

metric_managerB = SimpleManager([ospa_generator, siap_generator, uncertainty_generator],
                                associator=associator)

# %%
# For each time step, data is added to the metric manager on truths and tracks. The metrics themselves can then be
# generated from the metric manager.

metric_managerA.add_data(truths, tracksA)
metric_managerB.add_data(truths, tracksB)

metricsA = metric_managerA.generate_metrics()
metricsB = metric_managerB.generate_metrics()

# %%
# OSPA metric
# ^^^^^^^^^^^
#
# First we look at the OSPA metric. This is plotted over time for each sensor manager method.

import matplotlib.pyplot as plt

ospa_metricA = {metric for metric in metricsA if metric.title == "OSPA distances"}.pop()
ospa_metricB = {metric for metric in metricsB if metric.title == "OSPA distances"}.pop()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([i.timestamp for i in ospa_metricA.value],
        [i.value for i in ospa_metricA.value],
        label='TargetBasedRandomManager')
ax.plot([i.timestamp for i in ospa_metricB.value],
        [i.value for i in ospa_metricB.value],
        label='TargetBasedBruteForceManager')
ax.set_ylabel("OSPA distance")
ax.set_xlabel("Time")
ax.legend()

# %%
# The OSPA distnce for the :class:`TargetBasedBruteForceManager` is generally smaller than for the random observations
# of the :class:`TargetBasedRandomManager`.
#
# SIAP metrics
# ^^^^^^^^^^^^
#
# Next we look at SIAP metrics. This can be done by generating a table which displays all the SIAP metrics computed,
# as seen in the Metrics Example.
# 
# Completeness, ambiguity and spuriousness are not relevant for this example because we are not initiating and
# deleting tracks and we have one track corresponding to each ground truth.

fig, axes = plt.subplots(2)

for metric in metricsA:
    if metric.title.startswith('time-based SIAP PA'):
        pa_metricA = metric
    elif metric.title.startswith('time-based SIAP VA'):
        va_metricA = metric
    else:
        pass

for metric in metricsB:
    if metric.title.startswith('time-based SIAP PA'):
        pa_metricB = metric
    elif metric.title.startswith('time-based SIAP VA'):
        va_metricB = metric
    else:
        pass

times = metric_managerB.list_timestamps()

axes[0].set(title='Positional Accuracy', xlabel='Time', ylabel='PA')
axes[0].plot(times, [metric.value for metric in pa_metricA.value],
             label='TargetBasedRandomManager')
axes[0].plot(times, [metric.value for metric in pa_metricB.value],
             label='TargetBasedBruteForceManager')
axes[0].legend()

axes[1].set(title='Velocity Accuracy', xlabel='Time', ylabel='VA')
axes[1].plot(times, [metric.value for metric in va_metricA.value],
             label='TargetBasedRandomManager')
axes[1].plot(times, [metric.value for metric in va_metricB.value],
             label='TargetBasedBruteForceManager')
axes[1].legend()

# %%
# Similar to the OSPA distance the :class:`TargetBasedBruteForceManager` generally results in both a better
# positional accuracy and velocity accuracy than the random observations of the :class:`TargetBasedRandomManager`.
#
# Uncertainty metric
# ^^^^^^^^^^^^^^^^^^
#
# Finally we look at the uncertainty metric which computes the sum of covariance matrix norms of each state at each
# time step. This is plotted over time for each sensor manager method.

uncertainty_metricA = {metric for metric in metricsA if metric.title == "Sum of Covariance Norms Metric"}.pop()
uncertainty_metricB = {metric for metric in metricsB if metric.title == "Sum of Covariance Norms Metric"}.pop()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([i.timestamp for i in uncertainty_metricA.value],
        [i.value for i in uncertainty_metricA.value],
        label='TargetBasedRandomManager')
ax.plot([i.timestamp for i in uncertainty_metricB.value],
        [i.value for i in uncertainty_metricB.value],
        label='TargetBasedBruteForceManager')
ax.set_ylabel("Sum of covariance matrix norms")
ax.set_xlabel("Time")
ax.legend()

# sphinx_gallery_thumbnail_number = 6

# %%
# This metric shows that the uncertainty in the tracks generated by the :class:`TargetBasedRandomManager` is
# generally greater than for those generated by the :class:`TargetBasedBruteForceManager`.
# This is also reflected by the uncertainty ellipses
# in the initial plots of tracks and truths.
