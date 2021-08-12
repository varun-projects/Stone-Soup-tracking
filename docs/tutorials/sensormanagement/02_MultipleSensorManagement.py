#!/usr/bin/env python
# coding: utf-8

"""
==========================================================
2 - Multiple Sensor Management
==========================================================
"""

# %%
#
# This notebook follows on from the Single Sensor Management tutorial and further explores how existing
# Stone Soup features can be used to build simple sensor management algorithms for tracking and
# state estimation. This second tutorial demonstrates the limitations of the brute force optimisation
# method introduced in the previous tutorial by increasing the number of sensors used in the scenario.
#
# Introducing multiple sensors
# ----------------------------
# The example in this tutorial considers the same sensor management methods as in Tutorial 1 and applies them to the
# same set of ground truths in order to observe the difference in tracks. The scenario simulates 10
# targets moving on nearly constant velocity trajectories and in this case an adjustable number of sensors.
# The sensors are
# simple radar with a defined field of view which can be pointed in a particular direction in order
# to make an observation.
#
# The first method, using the class :class:`~.RandomSensorManager` chooses a target for each sensor to
# observe randomly with equal probability.
#
# The second method, uses the class :class:`~.BruteForceSensorManager` and aims to reduce the total
# uncertainty of the track estimates at each
# time step. To achieve this the sensor manager considers all possible configurations of directions for the sensors
# to point in. The sensor manager chooses the configuration for which the sum of estimated
# uncertainties (as represented by the Frobenius norm of the covariance matrix) can be reduced the most by observing
# the targets within the field of view when pointed in the chosen direction.
#
# The introduction of multiple sensors means an increase in the possible combinations of action choices
# that the brute force sensor manager must consider. This brute force optimisation method of looking at every possible
# combination of actions becomes very slow as more sensors are introduced. This demonstrates the
# limitations of using this method with a large number of sensors.
#
# As in the first tutorial the OSPA [#]_, SIAP [#]_ and uncertainty metrics are used to assess the performance of the
# sensor managers.

# %%
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
# Generate ground truths
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Generate transition model and ground truths as in Tutorial 1.
#
# The number of targets in this simulation is defined by `n_truths` - here there are 10 targets. The time the
# simulation is observed for is defined by `time_max`.
#
# We can fix our random number generator in order to probe a particular example repeatedly. This can be undone by
# commenting out the first two lines in the next cell.

np.random.seed(1990)
random.seed(1990)

# Generate transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])

yps = range(0, 120, 10)  # y value for prior state
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
# Create a set of sensors for each sensor management algorithm. As in Tutorial 1 this tutorial uses the
# :class:`~.SimpleRadar` sensor with the
# number of sensors initially set as 2. Each sensor is positioned along the line :math:`x=10`, at distance
# intervals of 50.
#
# Increasing the number of sensors above 2 significantly increases the run time of the sensor manager due to the
# increase in combinations to be considered by the :class:`~.BruteForceSensorManager`. This is discussed further later.

total_no_sensors = 2

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
# Create priors which estimate the targets' initial states - these are the same as in the first
# sensor management tutorial.

from stonesoup.types.state import GaussianState

priors = []
for j in range(0, ntruths):
    priors.append(GaussianState([[0], [1.5], [yps[j]+0.5], [1.5]], np.diag([1.5, 0.25, 1.5, 0.25]+np.random.normal(0,5e-4,4)),
                                timestamp=start_time))

# %%
# Initialise the tracks by creating an empty list and appending the priors generated. This needs to be done
# separately for both sensor manager methods as they will generate different sets of tracks.

from stonesoup.types.track import Track

# Initialise tracks from the RandomSensorManager
tracksA = []
for j, prior in enumerate(priors):
    tracksA.append(Track([prior], id=f"id{j}"))

# Initialise tracks from the BruteForceSensorManager
tracksB = []
for j, prior in enumerate(priors):
    tracksB.append(Track([prior], id=f"id{j}"))

# %%
# Create sensor managers
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Next we create our sensor management classes. As is Tutorial 1, two sensor manager classes are used -
# :class:`~.RandomSensorManager` and :class:`~.BruteForceSensorManager`.
#
# Random sensor manager
# """""""""""""""""""""
#
# The first method :class:`~.RandomSensorManager`, randomly chooses the action(s) for the sensors to
# take to make an observation. To do this the
# :meth:`choose_actions` function uses :meth:`random.choice()` to choose a direction for each
# sensor to observe from the possible actions it can take. It returns the chosen configuration of sensors and
# actions to be taken as a mapping.

from stonesoup.sensormanager import RandomSensorManager

# %%
# Brute force sensor manager
# """"""""""""""""""""""""""
#
# The second method :class:`~.BruteForceSensorManager` chooses the configuration of sensors and actions which results
# in the largest difference between the uncertainty covariances of the target predictions and posteriors
# assuming a predicted measurement corresponding to that prediction. This means the sensor manager chooses
# to point the sensors in directions such that the uncertainty will be reduced the most by
# making observations in those directions. This is evaluated by the reward function.

from stonesoup.sensormanager import BruteForceSensorManager

# %%
# Reward function
# """""""""""""""
# The :class:`RewardFunction` calculates the uncertainty reduction by computing the difference between the
# covariance matrix norms of the
# prediction and the posterior assuming a predicted measurement corresponding to that prediction. The sum
# of these differences is returned as a metric for that configuration.

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
# Create an instance of each sensor manager class. Both sensor managers take in a `sensor_set`.
# The :class:`~.BruteForceSensorManager` also requires a callable reward function which is initiated here
# from the :class:`RewardFunction` created above.


randomsensormanager = RandomSensorManager(sensor_setA)

# initiate reward function
reward_function = RewardFunction(predictor, updater)

bruteforcesensormanager = BruteForceSensorManager(sensor_setB,
                                                  reward_function=reward_function.calculate_reward)

# %%
# Run the sensor managers
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Both sensor management methods require a timestamp at each time step when calling
# the function :meth:`choose_actions` and the :class:`~.BruteForceSensorManager` also requires a list of tracks
# in order to evaluate the reward function. This returns a mapping of sensors and actions to be taken by each
# sensor, decided by the sensor managers.
#
# For both sensor management methods, at each time step the chosen action is given to the sensors and then
# measurements taken. The tracks are updated based on these measurements with predictions made for tracks
# which have not been observed.
#
# First a hypothesiser and data associator are required for use in both trackers.

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=5)

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
data_associator = GNNWith2DAssignment(hypothesiser)

# %%
# Run random sensor manager
# """""""""""""""""""""""""
#
# Here the chosen target for observation is selected randomly using the method :meth:`choose_actions()` from the class
# :class:`~.RandomSensorManager`. This returns a mapping of sensors to actions where actions are a direction for
# the sensor to point in, selected randomly.

from ordered_set import OrderedSet

# Generate list of timesteps from ground truth timestamps
timesteps = []
for state in truths[0]:
    timesteps.append(state.timestamp)

for timestep in timesteps[1:]:

    # Generate chosen configuration
    chosen_action = randomsensormanager.choose_actions(timestep)

    # Create empty dictionary for measurements
    measurementsA = []

    for sensor, actions in chosen_action.items():
        sensor.add_actions(actions)

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
# Plot ground truths, tracks and uncertainty ellipses for each target. The positions of the sensors are indicated
# by black x markers.

from matplotlib.lines import Line2D

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
# In comparison to Tutorial 1 the performance of the :class:`~.RandomSensorManager` has improved. This is
# because a greater number of sensors means each target is more likely to be observed. This means the uncertainty
# of the track does not increase as much because the targets are observed more often.

# %%
# Run brute force sensor manager
# """"""""""""""""""""""""""""""
#
# Here the direction for observation is selected based on the difference between the covariance matrices of the
# prediction and the update of predicted measurement, for targets which could be observed by the sensor
# pointing in the given direction.
#
# Within the sensor manager a dictionary is created of sensors and all the possible actions they can take.
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
    chosen_actions = bruteforcesensormanager.choose_actions(tracksB, timestep)

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
# The smaller uncertainty ellipses in this plot suggest that the :class:`~.BruteForceSensorManager` provides a much
# better track than the :class:`~.RandomSensorManager`.

# %%
# Combinatorics
# -------------
#
# The following graph demonstrates how the number of possible configurations increases with the number
# of sensors and number of targets. The number of configurations which are considered by the sensor manager
# for :math:`M` targets and :math:`N` sensors is :math:`M^N`.
#
# In this example there are 10 targets so the number of possible configurations should be :math:`10^N`
# where :math:`N` is the number of sensors. This exponential increase means that as larger number of
# sensors slows down the run time of the sensor manager significantly because there are so many more iterations
# to consider.
#
# Changing the number of sensors to :math:`N\geq 3` leads to a much longer run time.
# This highlights a practical limitation of using this brute force optimisation method for multiple
# sensors.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nsensors = np.arange(1, 100.0)
ntargets = np.arange(1, 100.0)
nsensors, ntargets = np.meshgrid(nsensors, ntargets)
ncombinations = ntargets**nsensors

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(nsensors, ntargets, np.log10(ncombinations), cmap='coolwarm')
ax.set_xlabel("No. sensors")
ax.set_ylabel("No. targets")
ax.set_zlabel("log of no. combinations")

# %%
# Metrics
# -------
#
# Metrics can be used to compare how well different sensor management techniques are working.
# As in Tutorial 1 the metrics used here are the OSPA, SIAP and uncertainty metrics.

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
# For each time step, data is added to the metric manager on truths and tracks.
# The metrics themselves can then be generated from the metric manager.

metric_managerA.add_data(truths, tracksA)
metric_managerB.add_data(truths, tracksB)

metricsA = metric_managerA.generate_metrics()
metricsB = metric_managerB.generate_metrics()

# %%
# OSPA metric
# ^^^^^^^^^^^
#
# First we look at the OSPA metric. This is plotted over time for each sensor manager method.

ospa_metricA = {metric for metric in metricsA if metric.title == "OSPA distances"}.pop()
ospa_metricB = {metric for metric in metricsB if metric.title == "OSPA distances"}.pop()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([i.timestamp for i in ospa_metricA.value],
        [i.value for i in ospa_metricA.value],
        label='RandomSensorManager')
ax.plot([i.timestamp for i in ospa_metricB.value],
        [i.value for i in ospa_metricB.value],
        label='BruteForceSensorManager')
ax.set_ylabel("OSPA distance")
ax.set_xlabel("Time")
ax.legend()

# %%
# The OSPA distance for the :class:`~.BruteForceSensorManager` is generally smaller than for the random
# observations of the :class:`~.RandomSensorManager`.
#
# SIAP metrics
# ^^^^^^^^^^^^
#
# Next we look at SIAP metrics. We are only interested in the positional accuracy (PA) and
# velocity accuracy (VA). These metrics can be plotted to show how they change over time.

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
             label='RandomSensorManager')
axes[0].plot(times, [metric.value for metric in pa_metricB.value],
             label='BruteForceSensorManager')
axes[0].legend()

axes[1].set(title='Velocity Accuracy', xlabel='Time', ylabel='VA')
axes[1].plot(times, [metric.value for metric in va_metricA.value],
             label='RandomSensorManager')
axes[1].plot(times, [metric.value for metric in va_metricB.value],
             label='BruteForceSensorManager')
axes[1].legend()

# %%
# Similar to the OSPA distance the :class:`~.BruteForceSensorManager` generally results in both a better
# positional accuracy and velocity accuracy than the random observations of the :class:`~.RandomSensorManager`.
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
        label='RandomSensorManager')
ax.plot([i.timestamp for i in uncertainty_metricB.value],
        [i.value for i in uncertainty_metricB.value],
        label='BruteForceSensorManager')
ax.set_ylabel("Sum of covariance matrix norms")
ax.set_xlabel("Time")
ax.legend()

# sphinx_gallery_thumbnail_number = 7

# %%
# This metric shows that the uncertainty in the tracks generated by the :class:`~.RandomSensorManager` is
# generally greater than for those generated by the :class:`~.BruteForceSensorManager`.
# This is also reflected by the uncertainty ellipses
# in the initial plots of tracks and truths.
#
# The uncertainty for the :class:`~.BruteForceSensorManager` starts poor and then improves initially as
# observations are made. This initial uncertainty is because the priors given are not correct. The uncertainty
# appears to increase in places. This is likely because a target has gone unobserved for too long and is
# no longer where the sensor thinks it is.

# %%
# References
# ----------
#
# .. [#] *D. Schuhmacher, B. Vo and B. Vo*, **A Consistent Metric for Performance Evaluation of
#    Multi-Object Filters**, IEEE Trans. Signal Processing 2008
# .. [#] *Votruba, Paul & Nisley, Rich & Rothrock, Ron and Zombro, Brett.*, **Single Integrated Air
#    Picture (SIAP) Metrics Implementation**, 2001

